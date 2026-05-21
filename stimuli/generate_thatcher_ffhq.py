"""stimuli/generate_thatcher_ffhq.py — Thatcher battery from FFHQ 1024×1024.

Reads FFHQ tar shards directly, runs MediaPipe at native 1024×1024, applies the
Thatcherize transform at native resolution, then saves PNGs (still 1024) plus
identity manifest and landmark JSONs in the SAME format as generate_thatcher.py
(so extract/compute_metrics scripts work unchanged).

Why high-res matters:
  - At 1024, the eye+brow region is ~150 px; Thatcher rotation produces a visually
    dramatic effect that humans agree is "grotesque" (canonical Thatcher illusion).
  - At LFW 224, the same region is ~30 px; rotation is barely visible to humans.
  - Model preprocessors auto-resize to their input size (typically 224); they
    receive the post-resize signal which still carries the local-feature inversion
    encoded at high res.

QC filters identical to generate_thatcher.py:
  - MediaPipe FaceMesh detection confidence ≥ 0.7
  - inter-ocular distance ≥ min_iod_px
  - eye-line tilt from horizontal ≤ 15 deg
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import random
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

# Reuse exact same Thatcherize implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stimuli.generate_thatcher import (
    landmarks_to_array,
    thatcherize,
    rotate_180,
    LEFT_EYE_REGION,
    RIGHT_EYE_REGION,
)

mp_fm = mp.solutions.face_mesh


@dataclass
class Identity:
    identity_id: int
    ffhq_name: str
    canonical_png_path: str
    landmarks_json_path: str


def iter_ffhq_shards(shard_dir: Path, shuffle: bool = True, seed: int = 20260521):
    """Yield (ffhq_id, PIL.Image) pairs across all tar shards in shard_dir."""
    tars = sorted(shard_dir.glob("*.tar"))
    if not tars:
        raise FileNotFoundError(f"no tar shards found in {shard_dir}")
    rng = random.Random(seed)
    for tar_path in tars:
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".webp")]
            if shuffle:
                rng.shuffle(members)
            for m in members:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue
                yield m.name.replace(".webp", ""), img


def main(args):
    shard_dir = Path(args.shard_dir)
    out_root = Path(args.output_root)
    identities_dir = out_root / "identities"
    landmarks_dir = out_root / "identity_landmarks"
    thatcher_dir = out_root / "thatcher"
    for d in (identities_dir, landmarks_dir, thatcher_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[init] MediaPipe FaceMesh refine_landmarks={args.refine}, target_n={args.target_n}")

    identity_records: list[Identity] = []
    manifest_rows: list[dict] = []

    with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                        refine_landmarks=args.refine,
                        min_detection_confidence=0.7) as mp_face:
        pbar = tqdm(iter_ffhq_shards(shard_dir, shuffle=True, seed=args.seed),
                    desc="scanning FFHQ", total=args.scan_cap)
        n_seen = 0
        n_rejected_landmarks = 0
        n_rejected_iod = 0
        n_rejected_tilt = 0
        for ffhq_name, pil_img in pbar:
            n_seen += 1
            if n_seen > args.scan_cap:
                break
            if len(identity_records) >= args.target_n:
                break
            rgb = np.array(pil_img)
            h, w = rgb.shape[:2]
            result = mp_face.process(rgb)
            if not result.multi_face_landmarks:
                n_rejected_landmarks += 1
                continue
            landmarks = result.multi_face_landmarks[0].landmark
            lm_arr = landmarks_to_array(landmarks, w, h)
            # QC: IoD
            try:
                le = lm_arr[LEFT_EYE_REGION].mean(axis=0)
                re = lm_arr[RIGHT_EYE_REGION].mean(axis=0)
            except IndexError:
                n_rejected_landmarks += 1
                continue
            iod = float(np.linalg.norm(le - re))
            if iod < args.min_iod_px:
                n_rejected_iod += 1
                continue
            # QC: eye-line tilt (absolute)
            dy = re[1] - le[1]
            dx = re[0] - le[0]
            tilt_deg = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
            if tilt_deg > args.max_tilt_deg:
                n_rejected_tilt += 1
                continue

            identity_id = len(identity_records)
            # Scale Thatcher padding with resolution
            eye_pad = max(8, int(min(h, w) * 0.02))      # ~20 px at 1024
            mouth_pad = max(10, int(min(h, w) * 0.025))  # ~25 px at 1024
            blur_sigma = max(3.0, min(h, w) * 0.005)     # ~5 px at 1024

            v1 = rgb.copy()
            v2 = thatcherize(rgb, lm_arr, eye_pad, mouth_pad, blur_sigma)
            v3 = rotate_180(rgb)
            v4 = rotate_180(v2)

            # Persist canonical identity image + landmarks JSON
            canonical_path = identities_dir / f"identity_{identity_id:04d}.png"
            Image.fromarray(rgb).save(canonical_path)
            lm_json_path = landmarks_dir / f"identity_{identity_id:04d}.json"
            with open(lm_json_path, "w") as f:
                json.dump({
                    "identity_id": identity_id,
                    "ffhq_name": ffhq_name,
                    "native_size": [int(w), int(h)],
                    "landmarks": [[float(x), float(y)] for x, y in lm_arr.tolist()],
                    "iod_px": iod,
                    "tilt_deg": tilt_deg,
                    "eye_pad_used": eye_pad,
                    "mouth_pad_used": mouth_pad,
                    "blur_sigma_used": blur_sigma,
                }, f)
            identity_records.append(Identity(
                identity_id=identity_id,
                ffhq_name=ffhq_name,
                canonical_png_path=str(canonical_path),
                landmarks_json_path=str(lm_json_path),
            ))

            for vname, vimg, orient, thatch_flag in [
                ("V1_upright_normal", v1, "upright", False),
                ("V2_upright_thatched", v2, "upright", True),
                ("V3_inverted_normal", v3, "inverted", False),
                ("V4_inverted_thatched", v4, "inverted", True),
            ]:
                path = thatcher_dir / f"identity_{identity_id:04d}_{vname}.png"
                Image.fromarray(vimg).save(path)
                manifest_rows.append({
                    "stim_id": f"thatcher_id{identity_id:04d}_{vname}",
                    "identity_id": identity_id,
                    "lfw_name": ffhq_name,
                    "paradigm": "thatcher",
                    "condition": vname,
                    "orientation": orient,
                    "thatcherized": int(thatch_flag),
                    "path": str(path),
                })

            pbar.set_postfix(accepted=len(identity_records),
                             nolm=n_rejected_landmarks,
                             iod=n_rejected_iod,
                             tilt=n_rejected_tilt)

    print(f"[done] {len(identity_records)} identities accepted "
          f"(scanned ~{n_seen}, reject: landmark={n_rejected_landmarks}, "
          f"iod={n_rejected_iod}, tilt={n_rejected_tilt})")

    # Note: column name kept as "lfw_name" for downstream compat; here it stores FFHQ id.
    manifest_path = out_root / "thatcher_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    identity_manifest_path = out_root / "identity_manifest.csv"
    with open(identity_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(identity_records[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in identity_records])

    print(f"  identity manifest: {identity_manifest_path}")
    print(f"  thatcher manifest: {manifest_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", required=True,
                   help="directory containing FFHQ webdataset .tar shards")
    p.add_argument("--output_root", required=True)
    p.add_argument("--target_n", type=int, default=200)
    p.add_argument("--scan_cap", type=int, default=1500,
                   help="hard cap on how many FFHQ images to scan")
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--min_iod_px", type=float, default=120.0,
                   help="MUCH higher at native 1024 vs 30 at LFW 224")
    p.add_argument("--max_tilt_deg", type=float, default=15.0)
    p.add_argument("--refine", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
