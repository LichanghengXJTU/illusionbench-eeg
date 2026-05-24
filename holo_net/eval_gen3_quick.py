"""holo_net/eval_gen3_quick.py — quick IllusionBench §6 eval on a Gen 3
backbone checkpoint, no FTPC head, just raw afp_pooled features.

For early-signal eval at intermediate training step (e.g. step 25K). Tells us
whether the Gen 3 backbone training has produced features with non-trivial
orientation-asymmetric processing, ahead of full training completion.
"""
from __future__ import annotations
import argparse
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

from holo_net.gen3_model import Gen3Backbone, Gen3ModelConfig

PARADIGMS = {
    "thatcher":   ("data/stimuli_ffhq/thatcher_manifest.csv",            1.000, "ISI"),
    "composite":  ("data/stimuli_ffhq_composite/thatcher_manifest.csv",  1.099, "CSI"),
    "partwhole":  ("data/stimuli_ffhq_partwhole/thatcher_manifest.csv",  1.202, "PWI"),
    "randombbox": ("data/stimuli_ffhq_randombbox/thatcher_manifest.csv", 1.000, "ISIrbox"),
}
CRITERIA = {"thatcher": (">=", 3.0), "composite": (">=", 1.5),
            "partwhole": ("<=", 0.5), "randombbox": ("<=", 1.5)}

IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


def make_eval_transform(size: int = 224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


def run_compute_metrics(emb_dir: Path, manifest: Path, output_csv: Path,
                          repo: Path) -> None:
    cmd = [sys.executable, "-m", "analysis.compute_metrics",
           "--embedding_dir", str(emb_dir),
           "--manifest", str(manifest),
           "--output", str(output_csv)]
    subprocess.run(cmd, check=True, cwd=str(repo),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(args.repo_root)

    print(f"[load] {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    step = ckpt.get("step", "?")
    epoch = ckpt.get("epoch", "?")
    print(f"  step={step}  epoch={epoch}")

    # Use student (or teacher — could compare both)
    which = args.which_weights
    mcfg = Gen3ModelConfig()
    model = Gen3Backbone(mcfg).to(device).eval()
    state = ckpt[which]
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"  loaded {which}; missing={len(missing)} unexpected={len(unexpected)}")

    tf = make_eval_transform()
    per_para = {}
    for para, (manifest_rel, pixbase, idx_name) in PARADIGMS.items():
        manifest = repo / manifest_rel
        df = pd.read_csv(manifest).sort_values("stim_id").reset_index(drop=True)
        stim_ids = df["stim_id"].to_numpy()
        print(f"\n[{para}] {len(df)} stims")
        imgs = [tf(Image.open(p).convert("RGB"))
                for p in tqdm(df["path"], desc=f"{para} load", leave=False)]
        bs = args.batch_size
        afp = []
        for i in tqdm(range(0, len(imgs), bs), desc=f"{para} forward", leave=False):
            x = torch.stack(imgs[i:i+bs]).to(device)
            out = model(x, mode="features")
            afp.append(out["afp_pooled"].cpu())
        afp = torch.cat(afp, dim=0)
        emb_n = F.normalize(afp, p=2, dim=1).numpy().astype(np.float32)
        emb_dir = repo / "outputs/embeddings" / f"{para}_{args.tag}"
        emb_dir.mkdir(parents=True, exist_ok=True)
        model_id = f"{args.tag}_afp"
        np.savez(emb_dir / f"{model_id}.npz",
                 stim_ids=stim_ids, embeddings=emb_n,
                 model_id=model_id, output_dim=int(emb_n.shape[1]))
        table = repo / "outputs/tables" / f"{args.tag}_{para}.csv"
        run_compute_metrics(emb_dir, manifest, table, repo)
        df_t = pd.read_csv(table)
        isi = float(df_t.iloc[0]["isi_pert"])
        isi_lo = float(df_t.iloc[0]["isi_pert_ci_lo"])
        isi_hi = float(df_t.iloc[0]["isi_pert_ci_hi"])
        per_para[para] = {"isi_pert": isi,
                          "ci_lo": isi_lo, "ci_hi": isi_hi,
                          "pixel_baseline": pixbase,
                          "pixel_corrected": isi / pixbase}

    # Headline table
    print(f"\n=== Gen 3 v4.0 step-{step} early IllusionBench §6 ===")
    print(f"{args.which_weights} weights, raw afp_pooled (NO FTPC head)")
    print()
    print("paradigm      ISI raw  pixel-corrected  CI [lo,hi]      §6 verdict")
    print("-" * 75)
    all_pass = True
    for p, (op, thr) in CRITERIA.items():
        pp = per_para[p]
        v = pp["pixel_corrected"]
        ok = (v >= thr) if op == ">=" else (v <= thr)
        all_pass = all_pass and ok
        print(f"{p:13s} {pp['isi_pert']:8.3f}  {v:8.3f}        "
              f"[{pp['ci_lo']:.3f}, {pp['ci_hi']:.3f}]  "
              f"need {op}{thr:.1f}  {'PASS' if ok else 'FAIL'}")
    print()
    print(f"ALL FOUR: {'*** PASS ***' if all_pass else 'FAIL'}")

    # Comparison to v3 (frozen DINOv2 raw afp) and Gen 1+2 best
    print()
    print("--- comparison to Gen 1+2 best ---")
    print("Gen 1+2 raw DINOv2 afp:  ISI 0.908  CSI 1.282  PWI 0.397  ISIrbox 0.662  (2/4)")
    print("Gen 1+2 best Gen 1 ffa:  ISI 1.082  CSI 1.328  PWI 0.594  ISIrbox 1.280  (1/4)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default="HOLONET_v4.0_gen3_25K")
    p.add_argument("--which_weights", default="student",
                   choices=["student", "teacher"])
    p.add_argument("--repo_root", default="/workspace/illusionbench-eeg")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
