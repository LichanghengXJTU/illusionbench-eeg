"""holo_net/eval_falsification_dinov2_dual.py — single-shot eval for v3.0
dual-template. Runs DINOv2 forward + concat readout + compute_metrics in one
pass per paradigm."""
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

from holo_net.model_v2_dinov2_dualtemplate import (
    HOLONetV2Dinov2Dual, HOLONetV2DinoDualConfig)

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


def make_eval_transform(size: int = INPUT_SIZE):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC,
                 antialias=True),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


def run_compute_metrics(emb_dir, manifest, output_csv, repo_root):
    cmd = [sys.executable, "-m", "analysis.compute_metrics",
           "--embedding_dir", str(emb_dir),
           "--manifest", str(manifest),
           "--output", str(output_csv)]
    subprocess.run(cmd, check=True, cwd=str(repo_root),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(args.repo_root)

    mcfg = HOLONetV2DinoDualConfig()
    model = HOLONetV2Dinov2Dual(mcfg).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] dual-template ckpt; missing={len(missing)} unexpected={len(unexpected)}")
    t_up = float(model.face_template_up.template.norm().item())
    t_inv = float(model.face_template_inv.template.norm().item())
    print(f"  T_up norm={t_up:.2f}  T_inv norm={t_inv:.2f}")

    tf = make_eval_transform()
    per_para_per_layer = {}
    for para, (manifest_rel, pixbase, idx_name) in PARADIGMS.items():
        manifest = repo / manifest_rel
        df = pd.read_csv(manifest).sort_values("stim_id").reset_index(drop=True)
        stim_ids = df["stim_id"].to_numpy()
        print(f"\n=== {para}: {len(df)} stimuli ===")
        imgs = [tf(Image.open(p).convert("RGB"))
                for p in tqdm(df["path"], desc=f"{para} load")]
        bs = args.batch_size
        feats = {"afp": [], "ffa": [], "delta_up": [], "delta_inv": []}
        for i in tqdm(range(0, len(imgs), bs), desc=f"{para} forward"):
            x = torch.stack(imgs[i:i+bs]).to(device)
            out = model(x)
            feats["afp"].append(out["afp_pooled"].cpu())
            feats["ffa"].append(out["ffa"].cpu())
            feats["delta_up"].append(out["delta_up"].cpu())
            feats["delta_inv"].append(out["delta_inv"].cpu())
        emb_dir = repo / "outputs/embeddings" / f"{para}_HOLONET_v3.0_dual"
        emb_dir.mkdir(parents=True, exist_ok=True)
        per_layer = {}
        for layer, fs in feats.items():
            emb = torch.cat(fs, dim=0)
            emb_n = F.normalize(emb, p=2, dim=1).numpy().astype(np.float32)
            model_id = f"HOLONET_v3.0_dual_{layer}"
            np.savez(emb_dir / f"{model_id}.npz",
                     stim_ids=stim_ids, embeddings=emb_n,
                     model_id=model_id, output_dim=int(emb_n.shape[1]))
        table = repo / "outputs/tables" / f"HOLONET_v3.0_dual_{para}.csv"
        run_compute_metrics(emb_dir, manifest, table, repo)
        df_t = pd.read_csv(table)
        for _, row in df_t.iterrows():
            layer = str(row["model_id"]).rsplit("_", 1)[-1]
            per_layer[layer] = float(row["isi_pert"]) / pixbase
        per_para_per_layer[para] = per_layer

    # Headline
    print(f"\n=== v3.0 dual-template falsification (pixel-corrected) ===")
    layers_print = ["afp", "ffa", "up", "inv"]
    header = "layer    " + "".join(f"{PARADIGMS[p][2]:>11}" for p in PARADIGMS) + "  pass/4"
    print(header)
    print("-" * len(header))
    for layer in layers_print:
        line = f"{layer:8s} "
        n_pass = 0
        for p in PARADIGMS:
            v = per_para_per_layer[p].get(layer, float("nan"))
            line += f"{v:11.3f}"
            op, thr = CRITERIA[p]
            ok = (v >= thr) if op == ">=" else (v <= thr)
            if ok:
                n_pass += 1
        line += f"   {n_pass}/4"
        print(line)

    # §6 verdict at FFA
    print("\n=== FFA verdict (= concat[δ_up, δ_inv] 768-d) ===")
    all_pass = True
    for p, (op, thr) in CRITERIA.items():
        v = per_para_per_layer[p].get("ffa", float("nan"))
        ok = (v >= thr) if op == ">=" else (v <= thr)
        all_pass = all_pass and ok
        print(f"  {PARADIGMS[p][2]:8s} = {v:7.3f}   need {op} {thr:<4}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\n  ALL FOUR (FFA): {'*** PASS ***' if all_pass else 'FAIL'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo_root", default="/workspace/illusionbench-eeg")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
