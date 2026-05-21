"""extract/extract_embeddings.py — sequential model forward pass over a stimulus set.

For each requested model in the registry, load it, run on all stimuli in the manifest,
save (stim_ids, embeddings) NPZ. Single-GPU H100 sequential loop.

Usage:
  python -m extract.extract_embeddings \
      --manifest data/stimuli/thatcher_manifest.csv \
      --output_dir outputs/embeddings/thatcher \
      --models P02_clip_b32,P04_clip_h14,P08_dinov2_large
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from models.registry import REGISTRY, free_model


def main(args):
    manifest = pd.read_csv(args.manifest)
    manifest = manifest.sort_values("stim_id").reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] preloading {len(manifest)} images")
    images = []
    for row in tqdm(manifest.itertuples(), total=len(manifest), desc="images"):
        img = Image.open(row.path).convert("RGB")
        images.append(img)
    stim_ids = manifest["stim_id"].to_numpy()

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_ids = list(REGISTRY.keys())

    for model_id in model_ids:
        if model_id not in REGISTRY:
            print(f"[skip] unknown model_id={model_id}")
            continue
        out_path = out_dir / f"{model_id}.npz"
        if out_path.exists() and not args.force:
            print(f"[skip] {model_id} already exists at {out_path}")
            continue

        print(f"[load] {model_id}")
        t0 = time.time()
        embed_fn, info, model_obj = REGISTRY[model_id]()
        t_load = time.time() - t0
        print(f"  loaded in {t_load:.1f}s  dim={info['output_dim']}  hf={info.get('hf_id')}")

        t0 = time.time()
        all_embeds = []
        bs = args.batch_size
        for i in tqdm(range(0, len(images), bs), desc=f"embed {model_id}"):
            batch = images[i:i + bs]
            emb = embed_fn(batch)
            all_embeds.append(emb)
        embeddings = np.concatenate(all_embeds, axis=0).astype(np.float32)
        t_embed = time.time() - t0

        np.savez(
            out_path,
            stim_ids=stim_ids,
            embeddings=embeddings,
            model_id=model_id,
            hf_id=str(info.get("hf_id", "")),
            output_dim=int(info["output_dim"]),
        )
        print(f"[save] {out_path}  shape={embeddings.shape}  embed_time={t_embed:.1f}s")

        free_model(model_obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", default="", help="comma-separated model IDs (default: all in registry)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
