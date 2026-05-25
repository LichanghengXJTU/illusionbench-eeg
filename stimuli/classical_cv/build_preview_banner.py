"""stimuli/classical_cv/build_preview_banner.py — make HF dataset banner.

Picks K diverse identities from the parallel-Thatcher output dir and tiles
them into a 4-column (V1/V2/V3/V4) × K-row contact sheet for use as the
README banner on Hugging Face.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main(args):
    src_root = Path(args.input_dir)
    ids = sorted(src_root.glob("ffhq_*"))
    if len(ids) < args.k:
        raise SystemExit(f"need ≥ {args.k} identity dirs; found {len(ids)}")

    rng = random.Random(args.seed)
    # Pick K identities evenly spread across the FFHQ index range for diversity
    step = max(1, len(ids) // (args.k * 2))
    candidates = ids[::step]
    rng.shuffle(candidates)
    picked = candidates[:args.k]
    print(f"[init] picked {args.k} identities (from {len(ids)} total):")
    for p in picked:
        print(f"  {p.name}")

    tile = args.tile_size
    header = 24
    label_strip = 26
    cols = 4
    rows = args.k
    canvas = np.full((rows * tile + header + label_strip,
                       cols * tile, 3), 250, dtype=np.uint8)

    # Top header: condition labels
    cond_labels = ["V1 upright_normal",
                   "V2 upright_thatched",
                   "V3 inverted_normal",
                   "V4 inverted_thatched"]
    for j, lbl in enumerate(cond_labels):
        cv2.putText(canvas, lbl, (j * tile + 10, header - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    for i, idir in enumerate(picked):
        for j, cond in enumerate(["V1", "V2", "V3", "V4"]):
            img_path = idir / f"thatcher_{cond}.png"
            if not img_path.exists():
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            small = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            y0 = header + i * tile
            x0 = j * tile
            canvas[y0:y0 + tile, x0:x0 + tile] = small

    # Bottom label
    bottom_y = rows * tile + header + 20
    cv2.putText(canvas, "HoloFaceIllusion-Bench-EEG v1.0 (Thatcher) - "
                "sample identities; 26,317 total identities in release",
                (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 60, 60), 1, cv2.LINE_AA)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    print(f"\n[done] banner saved: {out_path} ({canvas.shape})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",
                   default="/workspace/illusionbench-eeg/data/thatcher_ffhq_70k_parallel")
    p.add_argument("--output", default="data/HoloFaceIllusion-Bench-EEG-v1.0/preview.png")
    p.add_argument("--k", type=int, default=6, help="rows of identities")
    p.add_argument("--tile_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
