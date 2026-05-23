"""holo_net/eval_v24_sweep.py — v2.4 sweep: weighted concat α + face-restricted
orient, in a single pass per paradigm.

Computes the following ffa variants:
  v2.4a_alpha0.00 — concat(δ_global, 0.00 × δ_orient) = δ_global only (= v3, sanity)
  v2.4a_alpha0.25 — concat(δ_global, 0.25 × δ_orient)
  v2.4a_alpha0.50 — concat(δ_global, 0.50 × δ_orient)
  v2.4a_alpha0.75 — concat(δ_global, 0.75 × δ_orient)
  v2.4a_alpha1.00 — concat(δ_global, 1.00 × δ_orient) = v2.3 (sanity)
  v2.4a_alpha2.00 — concat(δ_global, 2.00 × δ_orient)
  v2.4b_face — concat(δ_global, concat over K parts of (AFP_vflip[region_k] - T_part_k)_pooled)

Reuses v3 ckpt for T_global; v2.2 ckpt for T_part_k.

Single forward over each stimulus set (forward x and vflip(x) through frozen
DINOv2; cache spatial outputs; compose ffa variants in pure tensor ops).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from tqdm import tqdm

from holo_net.model_v2_dinov2_partaware import PART_REGIONS

PARADIGMS = {
    "thatcher":   ("data/stimuli_ffhq/thatcher_manifest.csv",            1.000, "ISI"),
    "composite":  ("data/stimuli_ffhq_composite/thatcher_manifest.csv",  1.099, "CSI"),
    "partwhole":  ("data/stimuli_ffhq_partwhole/thatcher_manifest.csv",  1.202, "PWI"),
    "randombbox": ("data/stimuli_ffhq_randombbox/thatcher_manifest.csv", 1.000, "ISIrbox"),
}
CRITERIA = {"thatcher": (">=", 3.0), "composite": (">=", 1.5),
            "partwhole": ("<=", 0.5), "randombbox": ("<=", 1.5)}

INPUT_SIZE = 224
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)
PARTS = ("eyes", "nose", "mouth", "chin")


def make_eval_transform(size: int = INPUT_SIZE):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC,
                 antialias=True),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


def load_dinov2(torch_cache: str = "/workspace/.torch_cache") -> torch.nn.Module:
    os.environ.setdefault("TORCH_HOME", torch_cache)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                            trust_repo=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_patch_tokens(backbone, x: torch.Tensor) -> torch.Tensor:
    out = backbone.forward_features(x)
    patch = out["x_norm_patchtokens"]
    B, N, D = patch.shape
    H = W = int(N ** 0.5)
    return patch.transpose(1, 2).reshape(B, D, H, W).contiguous()


def load_T_global(ckpt_path: Path) -> torch.Tensor:
    """Load the v3 global face_template (D, H, W) from its checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    T = state["face_template.template"]   # (D, H, W)
    return T


def load_T_parts(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load the v2.2 K part_templates (each shape D, h_k, w_k)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    parts = {}
    for k in PARTS:
        key = f"part_templates.{k}.template"
        parts[k] = state[key]
    return parts


def normalize_save(emb: torch.Tensor, stim_ids: np.ndarray,
                    model_id: str, out_dir: Path) -> Path:
    emb_n = F.normalize(emb, p=2, dim=1).cpu().numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{model_id}.npz"
    np.savez(out_npz, stim_ids=stim_ids, embeddings=emb_n,
             model_id=model_id, output_dim=int(emb_n.shape[1]))
    return out_npz


def run_compute_metrics(emb_dir: Path, manifest: Path, output_csv: Path,
                          repo_root: Path) -> None:
    cmd = [sys.executable, "-m", "analysis.compute_metrics",
           "--embedding_dir", str(emb_dir),
           "--manifest", str(manifest),
           "--output", str(output_csv)]
    subprocess.run(cmd, check=True, cwd=str(repo_root),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(args.repo_root)

    print(f"[device] {device}")
    print(f"[ckpt-global] {args.global_ckpt}")
    print(f"[ckpt-parts ] {args.parts_ckpt}")

    backbone = load_dinov2().to(device).eval()
    T_global = load_T_global(Path(args.global_ckpt)).to(device)             # (D, H, W)
    T_global_pooled = F.adaptive_avg_pool2d(T_global[None], 1).flatten(1).squeeze(0)  # (D,)
    T_parts = {k: v.to(device) for k, v in load_T_parts(Path(args.parts_ckpt)).items()}
    T_parts_pooled = {k: F.adaptive_avg_pool2d(v[None], 1).flatten(1).squeeze(0)
                      for k, v in T_parts.items()}
    for k in PARTS:
        print(f"[T_part {k}] norm={T_parts[k].norm().item():.2f}  "
              f"shape={tuple(T_parts[k].shape)}")
    print(f"[T_global] norm={T_global.norm().item():.2f}  "
          f"shape={tuple(T_global.shape)}")

    # Variants to compute
    alphas = [0.00, 0.25, 0.50, 0.75, 1.00, 2.00]
    variant_tags = (
        [f"v2.4a_alpha{a:.2f}" for a in alphas] + ["v2.4b_face"]
    )
    per_para_per_var: dict[str, dict[str, dict[str, float]]] = {}

    tf = make_eval_transform()

    for para, (manifest_rel, pixbase, idx_name) in PARADIGMS.items():
        manifest = repo / manifest_rel
        df = pd.read_csv(manifest).sort_values("stim_id").reset_index(drop=True)
        stim_ids = df["stim_id"].to_numpy()
        print(f"\n=== {para}: {len(df)} stimuli ===")
        imgs = [tf(Image.open(p).convert("RGB"))
                for p in tqdm(df["path"], desc=f"{para} load")]

        # Compute AFP_pooled and AFP_vflip_spatial for all stimuli
        bs = args.batch_size
        afp_pool_list = []
        afp_vflip_pool_list = []
        afp_vflip_sp_list = []
        for i in tqdm(range(0, len(imgs), bs), desc=f"{para} forward"):
            x = torch.stack(imgs[i:i+bs]).to(device)
            afp_sp = extract_patch_tokens(backbone, x)                          # (B, D, H, W)
            afp_pool_list.append(F.adaptive_avg_pool2d(afp_sp, 1).flatten(1).cpu())
            x_vflip = torch.flip(x, dims=[-2])
            afp_vflip_sp = extract_patch_tokens(backbone, x_vflip)
            afp_vflip_pool_list.append(F.adaptive_avg_pool2d(afp_vflip_sp, 1).flatten(1).cpu())
            afp_vflip_sp_list.append(afp_vflip_sp.cpu())
        afp_pool = torch.cat(afp_pool_list, dim=0)              # (N, D)
        afp_vflip_pool = torch.cat(afp_vflip_pool_list, dim=0)   # (N, D)
        afp_vflip_sp = torch.cat(afp_vflip_sp_list, dim=0)       # (N, D, H, W)

        delta_global = afp_pool - T_global_pooled[None].cpu()    # (N, D)
        delta_orient = afp_vflip_pool - T_global_pooled[None].cpu()

        # Per-part delta_face_orient via v2.2 part regions
        delta_face_orient_parts = []
        for k in PARTS:
            r0, r1, c0, c1 = PART_REGIONS[k]
            region_vflip = afp_vflip_sp[..., r0:r1, c0:c1]                # (N, D, h, w)
            region_pooled = F.adaptive_avg_pool2d(region_vflip, 1).flatten(1)  # (N, D)
            delta_face_orient_parts.append(region_pooled - T_parts_pooled[k][None].cpu())
        delta_face_orient = torch.cat(delta_face_orient_parts, dim=1)    # (N, D × K)

        per_para_per_var[para] = {}
        for var_tag in variant_tags:
            if var_tag.startswith("v2.4a_alpha"):
                a = float(var_tag.split("alpha")[-1])
                ffa = torch.cat([delta_global, a * delta_orient], dim=1)  # (N, 2D)
            elif var_tag == "v2.4b_face":
                ffa = torch.cat([delta_global, delta_face_orient], dim=1)  # (N, (1+K)*D)
            else:
                raise ValueError(var_tag)

            emb_dir = repo / "outputs/embeddings" / f"{para}_HOLONET_{var_tag}"
            table = repo / "outputs/tables" / f"HOLONET_{var_tag}_{para}.csv"
            model_id = f"HOLONET_{var_tag}_afp"   # use "afp" suffix so compute_metrics has a single row
            normalize_save(ffa, stim_ids, model_id, emb_dir)
            run_compute_metrics(emb_dir, manifest, table, repo)
            row = pd.read_csv(table)
            isi = float(row.iloc[0]["isi_pert"])
            isi_lo = float(row.iloc[0]["isi_pert_ci_lo"])
            isi_hi = float(row.iloc[0]["isi_pert_ci_hi"])
            per_para_per_var[para][var_tag] = {
                "isi_pert": isi, "ci_lo": isi_lo, "ci_hi": isi_hi,
                "pixel_baseline": pixbase,
                "pixel_corrected": isi / pixbase,
            }

    # Headline table
    print(f"\n=== v2.4 sweep §6 verdict table (pixel-corrected) ===")
    header = "variant            " + "".join(f"{PARADIGMS[p][2]:>11}" for p in PARADIGMS) + "  pass/4"
    print(header)
    print("-" * len(header))
    for var_tag in variant_tags:
        line = f"{var_tag:18s} "
        n_pass = 0
        for p in PARADIGMS:
            v = per_para_per_var[p][var_tag]["pixel_corrected"]
            line += f"{v:11.3f}"
            op, thr = CRITERIA[p]
            ok = (v >= thr) if op == ">=" else (v <= thr)
            if ok:
                n_pass += 1
        line += f"   {n_pass}/4"
        print(line)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--global_ckpt",
                   default="/workspace/runs/2026-05-23_holonet-v2-dinov2_seed20260521/checkpoint.pt")
    p.add_argument("--parts_ckpt",
                   default="/workspace/runs/2026-05-23_holonet-v2.2-partaware_seed20260521/checkpoint.pt")
    p.add_argument("--repo_root", default="/workspace/illusionbench-eeg")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
