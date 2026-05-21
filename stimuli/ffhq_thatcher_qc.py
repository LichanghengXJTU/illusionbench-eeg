"""stimuli/ffhq_thatcher_qc.py — high-res Thatcher QC on FFHQ samples.

Extract N images from an FFHQ webdataset tar shard, apply Thatcherize at native
1024×1024 resolution, save 4 conditions per identity + a single contact sheet.

The visual purpose: confirm that OUR Thatcherize algorithm produces dramatic, classical
Thatcher illusion when given high-quality frontal portrait input. If yes → metric is
likely fine, LFW quality is the bottleneck. If no → algorithm has a problem.
"""
from __future__ import annotations
import argparse
import io
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2

# Reuse the Thatcherize implementation from generate_thatcher
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stimuli.generate_thatcher import (
    landmarks_to_array,
    thatcherize,
    rotate_180,
    LEFT_EYE_REGION,
    RIGHT_EYE_REGION,
)


def extract_first_n_webp(tar_path: Path, n: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".webp")]
        members = sorted(members, key=lambda m: m.name)[:n]
        for m in members:
            f = tf.extractfile(m)
            data = f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            out_path = out_dir / m.name.replace(".webp", ".png")
            img.save(out_path)
            extracted.append(out_path)
    return extracted


def main(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    src_dir = out_root / "ffhq_originals"
    stim_dir = out_root / "ffhq_thatcher_hires"
    src_dir.mkdir(exist_ok=True)
    stim_dir.mkdir(exist_ok=True)

    tar_path = Path(args.tar)
    print(f"[extract] {args.n} originals from {tar_path}")
    src_paths = extract_first_n_webp(tar_path, args.n, src_dir)

    mp_fm = mp.solutions.face_mesh
    print(f"[gen] running MediaPipe + Thatcherize at native resolution")
    pairs = []  # (ident_id, original_path, v1, v2, v3, v4)
    with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                        refine_landmarks=True, min_detection_confidence=0.7) as mp_face:
        for i, p in enumerate(src_paths):
            bgr = cv2.imread(str(p))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            result = mp_face.process(rgb)
            if not result.multi_face_landmarks:
                print(f"  [skip] no landmarks: {p.name}")
                continue
            lm = result.multi_face_landmarks[0].landmark
            lm_arr = landmarks_to_array(lm, w, h)
            # Scale Thatcher padding with resolution
            eye_pad = max(8, int(min(h, w) * 0.02))      # ~20 px at 1024
            mouth_pad = max(10, int(min(h, w) * 0.025))  # ~25 px at 1024
            blur_sigma = max(3.0, min(h, w) * 0.005)     # ~5 px at 1024

            v1 = rgb.copy()
            v2 = thatcherize(rgb, lm_arr, eye_pad, mouth_pad, blur_sigma)
            v3 = rotate_180(rgb)
            v4 = rotate_180(v2)
            stem = p.stem
            pid = f"{stem}_id{i:03d}"
            for vname, vimg in [("V1_upright_normal", v1),
                                ("V2_upright_thatched", v2),
                                ("V3_inverted_normal", v3),
                                ("V4_inverted_thatched", v4)]:
                Image.fromarray(vimg).save(stim_dir / f"{pid}_{vname}.png")
            pairs.append((pid, p, v1, v2, v3, v4))
            print(f"  [done] {pid}  ({h}×{w}, eye_pad={eye_pad}, mouth_pad={mouth_pad})")

    # Contact sheet at full resolution
    n = len(pairs)
    print(f"[contact] composing {n}-row sheet")
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), squeeze=False)
    titles = ["V1 upright_normal", "V2 upright_thatched", "V3 inverted_normal", "V4 inverted_thatched"]
    for r, (pid, _, v1, v2, v3, v4) in enumerate(pairs):
        for c, img in enumerate([v1, v2, v3, v4]):
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(pid, fontsize=9)
    plt.suptitle("FFHQ 1024×1024 — Thatcher sanity check", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    contact_path = out_root / "ffhq_thatcher_contact.png"
    fig.savefig(contact_path, dpi=80, bbox_inches="tight")
    print(f"[done] contact sheet: {contact_path}  ({contact_path.stat().st_size/1024/1024:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--n", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
