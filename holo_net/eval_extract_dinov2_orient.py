"""holo_net/eval_extract_dinov2_orient.py — per-layer NPZ extraction for
HOLO-Net v2.3 orientation-aware FTPC."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from tqdm import tqdm

from holo_net.model_v2_dinov2_orient import (
    HOLONetV2Dinov2Orient, HOLONetV2DinoOrientConfig)

LAYERS = ["v1", "v2", "v4", "mfp", "afp", "ffa", "atl"]
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


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = HOLONetV2DinoOrientConfig()
    model = HOLONetV2Dinov2Orient(mcfg).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_face = ckpt.get("n_faces_total", "?") if isinstance(ckpt, dict) else "?"
    print(f"[load] dinov2-orient ckpt (reusing v3 template)  n_faces_total={n_face}  device={device}")
    if missing:
        print(f"  [warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    t_norm = float(model.face_template.template.norm().item())
    print(f"  T norm={t_norm:.2f}")

    manifest = pd.read_csv(args.manifest).sort_values("stim_id").reset_index(drop=True)
    stim_ids = manifest["stim_id"].to_numpy()
    print(f"[stim] {len(manifest)} stimuli")
    tf = make_eval_transform()
    images = [tf(Image.open(p).convert("RGB"))
              for p in tqdm(manifest["path"], desc="load")]

    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    feats: dict[str, list] = {ln: [] for ln in layers}
    bs = args.batch_size
    for i in tqdm(range(0, len(images), bs), desc="forward"):
        x = torch.stack(images[i:i + bs]).to(device)
        out = model(x)
        for ln in layers:
            f = model.get_layer_for_eeg_readout(ln, out)
            feats[ln].append(f.float().cpu())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ln in layers:
        emb = torch.cat(feats[ln], dim=0)
        emb = F.normalize(emb, p=2, dim=1).numpy().astype(np.float32)
        model_id = f"{args.tag}_{ln}"
        np.savez(out_dir / f"{model_id}.npz",
                 stim_ids=stim_ids, embeddings=emb,
                 model_id=model_id, output_dim=int(emb.shape[1]))
        print(f"[save] {model_id}.npz  shape={emb.shape}")
    print(f"\n[done] {len(layers)} layer NPZ in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
