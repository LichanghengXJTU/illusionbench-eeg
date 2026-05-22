"""holo_net/eval_extract.py — extract per-brain-region embeddings from a trained
HOLO-Net checkpoint over an IllusionBench stimulus set.

Produces one NPZ per layer in the repo-standard schema (stim_ids, embeddings),
so the existing analysis/compute_metrics.py (and the composite / part-whole
metric scripts) consume them with NO changes.

Layers: v1, v2, v4, mfp, afp, ffa, atl — the design-doc brain-region readouts.
The HOLO-Net pre-registered falsification criteria (design doc §6) are read at
the FFA layer; the other layers give the per-layer dissociation profile.

Usage:
  python -m holo_net.eval_extract \
      --checkpoint /workspace/holo_net_full_v1/checkpoint.pt \
      --manifest   <thatcher_manifest.csv> \
      --output_dir <outputs/embeddings/thatcher_ffhq> \
      --tag HOLONET_full_v1
  # add --minimal if the checkpoint was trained with train.py --minimal
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from holo_net.model import HOLONet, HOLONetConfig

LAYERS = ["v1", "v2", "v4", "mfp", "afp", "ffa", "atl"]
INPUT_SIZE = 224


def build_config(minimal: bool) -> HOLONetConfig:
    """Mirror train.setup_model_and_loss so the checkpoint state_dict loads
    into a model with the matching set of components."""
    cfg = HOLONetConfig()
    if minimal:
        cfg.use_magno = False
        cfg.use_ofa_branch = False
        cfg.use_orientation_gate = False
        cfg.use_pfc_gist = False
        cfg.use_ffa = False
        cfg.use_pc_feedback = False
    return cfg


def preprocess(img: Image.Image) -> torch.Tensor:
    """Deterministic eval transform — data._augment_face minus the random
    augmentations: resize 224, scale to [0,1], normalize to [-1,1]."""
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)  # (H,W,3)
    x = x.permute(2, 0, 1)             # (3,H,W)
    x = (x - 0.5) / 0.5                # [0,1] → [-1,1]  (mean/std = 0.5)
    return x


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model ---
    cfg = build_config(args.minimal)
    model = HOLONet(cfg).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    print(f"[load] checkpoint step={step}  minimal={args.minimal}  device={device}")
    if missing:
        print(f"  [warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    # --- stimuli ---
    manifest = pd.read_csv(args.manifest).sort_values("stim_id").reset_index(drop=True)
    stim_ids = manifest["stim_id"].to_numpy()
    print(f"[stim] {len(manifest)} stimuli from {args.manifest}")
    images = [preprocess(Image.open(p)) for p in tqdm(manifest["path"], desc="load")]

    # --- forward, collect per-layer features ---
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    feats: dict[str, list] = {ln: [] for ln in layers}
    bs = args.batch_size
    for i in tqdm(range(0, len(images), bs), desc="forward"):
        x = torch.stack(images[i:i + bs]).to(device)
        out = model(x, return_all=True)
        for ln in layers:
            f = model.get_layer_for_eeg_readout(ln, out)  # (B, D)
            feats[ln].append(f.float().cpu())

    # --- save one NPZ per layer, repo-standard schema ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ln in layers:
        emb = torch.cat(feats[ln], dim=0)
        emb = F.normalize(emb, p=2, dim=1).numpy().astype(np.float32)  # cosine-ready
        model_id = f"{args.tag}_{ln}"
        np.savez(out_dir / f"{model_id}.npz",
                 stim_ids=stim_ids, embeddings=emb,
                 model_id=model_id, output_dim=int(emb.shape[1]))
        print(f"[save] {model_id}.npz  shape={emb.shape}")
    print(f"\n[done] {len(layers)} layer NPZ files in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True, help="CSV with stim_id, path columns")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tag", required=True, help="e.g. HOLONET_full_v1 — NPZ named <tag>_<layer>")
    p.add_argument("--minimal", action="store_true",
                   help="set if the checkpoint was trained with train.py --minimal")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
