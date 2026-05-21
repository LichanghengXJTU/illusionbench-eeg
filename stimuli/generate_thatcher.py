"""generate_thatcher.py — Thatcher illusion stimulus battery generator.

Given LFW funneled images, samples N identities and generates 4 stimuli per identity:
  V1 upright_normal      — the original face
  V2 upright_thatched    — eyes (L, R) and mouth regions each rotated 180 deg in-plane
  V3 inverted_normal     — full image rotated 180 deg
  V4 inverted_thatched   — V2 rotated 180 deg

Standard Thatcher psychophysics expectation:
  - Humans easily distinguish V1 vs V2 (upright thatcher looks grotesque)
  - Humans struggle to distinguish V3 vs V4 (inverted thatcher looks normal)

Side effect: this script also writes one-time identity selection artifacts under
`<output_root>/identity_landmarks/{identity_id:04d}.json` and the canonical normalized
face crop under `<output_root>/identities/identity_{identity_id:04d}.png`. Other
paradigm generators (composite, part-whole, configural) load these to avoid re-detection.

Usage:
  python generate_thatcher.py --lfw_dir /workspace/illusionbench-eeg/data/lfw_funneled \
                              --output_root /workspace/illusionbench-eeg/data/stimuli \
                              --target_n 200 --seed 20260521
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tqdm import tqdm

mp_fm = mp.solutions.face_mesh


def _connection_points(connections) -> list[int]:
    pts: set[int] = set()
    for a, b in connections:
        pts.add(a)
        pts.add(b)
    return sorted(pts)


# MediaPipe FaceMesh landmark groups derived from the official FACEMESH_* connection sets.
# For the Thatcher transform we extend each "eye" region to include the eyebrow:
# the classical illusion (Thompson 1980) flips the entire eye-eyebrow complex.
LEFT_EYE_PTS = _connection_points(mp_fm.FACEMESH_LEFT_EYE)
RIGHT_EYE_PTS = _connection_points(mp_fm.FACEMESH_RIGHT_EYE)
LEFT_BROW_PTS = _connection_points(mp_fm.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW_PTS = _connection_points(mp_fm.FACEMESH_RIGHT_EYEBROW)
# Eye-region for Thatcher: eye outline ∪ eyebrow points (so the bbox covers both).
LEFT_EYE_REGION = sorted(set(LEFT_EYE_PTS) | set(LEFT_BROW_PTS))
RIGHT_EYE_REGION = sorted(set(RIGHT_EYE_PTS) | set(RIGHT_BROW_PTS))
LIPS_PTS = _connection_points(mp_fm.FACEMESH_LIPS)


@dataclass
class Identity:
    identity_id: int
    lfw_name: str
    lfw_source_path: str
    canonical_png_path: str
    landmarks_json_path: str


def landmarks_to_array(landmarks, img_w: int, img_h: int) -> np.ndarray:
    """Convert MediaPipe normalized landmark list to (N, 2) pixel-coord array."""
    return np.array([[lm.x * img_w, lm.y * img_h] for lm in landmarks], dtype=np.float32)


def get_bbox(landmark_arr: np.ndarray, indices: Sequence[int],
             img_w: int, img_h: int, padding: int) -> tuple[int, int, int, int]:
    pts = landmark_arr[list(indices)]
    x1 = max(0, int(pts[:, 0].min() - padding))
    y1 = max(0, int(pts[:, 1].min() - padding))
    x2 = min(img_w, int(pts[:, 0].max() + padding))
    y2 = min(img_h, int(pts[:, 1].max() + padding))
    return x1, y1, x2, y2


def soft_alpha_mask(h: int, w: int, blur_sigma: float) -> np.ndarray:
    """Soft alpha mask: 1 in the interior, smoothly fading to 0 at the boundary."""
    pad = int(np.ceil(blur_sigma * 3))
    pad = max(1, min(pad, min(h, w) // 2 - 1))
    inner = np.zeros((h, w), dtype=np.float32)
    inner[pad:-pad, pad:-pad] = 1.0
    if blur_sigma > 0:
        inner = cv2.GaussianBlur(inner, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.clip(inner, 0.0, 1.0)


def paste_with_alpha(canvas: np.ndarray, patch: np.ndarray,
                     bbox: tuple[int, int, int, int], blur_sigma: float) -> None:
    """In-place: blend `patch` into `canvas[bbox]` with a soft boundary."""
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return
    alpha = soft_alpha_mask(h, w, blur_sigma)[..., None]
    base = canvas[y1:y2, x1:x2].astype(np.float32)
    new = patch.astype(np.float32)
    blended = base * (1.0 - alpha) + new * alpha
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


def thatcherize(img_rgb: np.ndarray, landmark_arr: np.ndarray,
                eye_pad: int, mouth_pad: int, blur_sigma: float) -> np.ndarray:
    """Rotate left eye+brow, right eye+brow, and mouth regions each 180 deg in-place."""
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    for region_indices, pad in [
        (LEFT_EYE_REGION, eye_pad),
        (RIGHT_EYE_REGION, eye_pad),
        (LIPS_PTS, mouth_pad),
    ]:
        bbox = get_bbox(landmark_arr, region_indices, w, h, pad)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        patch = result[y1:y2, x1:x2].copy()
        patch_rot = cv2.rotate(patch, cv2.ROTATE_180)
        paste_with_alpha(result, patch_rot, bbox, blur_sigma)
    return result


def rotate_180(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.rotate(img_rgb, cv2.ROTATE_180)


def select_identities(lfw_root: Path, mp_face, target_n: int, target_size: int,
                      seed: int, min_iod_px: float) -> list[dict]:
    """Walk LFW funneled, run MediaPipe, keep identities passing QC."""
    rng = random.Random(seed)
    person_dirs = sorted([d for d in lfw_root.iterdir() if d.is_dir()])
    rng.shuffle(person_dirs)

    accepted: list[dict] = []
    rejected_count = 0
    pbar = tqdm(person_dirs, desc="scanning LFW")
    for pdir in pbar:
        if len(accepted) >= target_n:
            break
        imgs = sorted(pdir.glob("*.jpg"))
        if not imgs:
            continue
        img_path = imgs[0]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            rejected_count += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Resize to canonical target_size while preserving the LFW funneled centering.
        rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        result = mp_face.process(rgb)
        if not result.multi_face_landmarks:
            rejected_count += 1
            continue
        landmarks = result.multi_face_landmarks[0].landmark
        lm_arr = landmarks_to_array(landmarks, target_size, target_size)
        # QC: inter-ocular distance large enough
        try:
            left_eye_center = lm_arr[LEFT_EYE_PTS].mean(axis=0)
            right_eye_center = lm_arr[RIGHT_EYE_PTS].mean(axis=0)
        except IndexError:
            rejected_count += 1
            continue
        iod = float(np.linalg.norm(left_eye_center - right_eye_center))
        if iod < min_iod_px:
            rejected_count += 1
            continue
        # QC: eye-line tilt from horizontal within ±15 deg.
        # We use abs(dy)/abs(dx) to be invariant to which eye MediaPipe labels "left"
        # (its LEFT/RIGHT are subject-frame; absolute angle from horizontal is what we need).
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        tilt_deg = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
        if tilt_deg > 15.0:
            rejected_count += 1
            continue
        accepted.append({
            "lfw_name": pdir.name,
            "lfw_source_path": str(img_path),
            "rgb": rgb,
            "landmarks": landmarks,
            "lm_arr": lm_arr,
            "iod_px": iod,
            "tilt_deg": tilt_deg,
        })
        pbar.set_postfix(accepted=len(accepted), rejected=rejected_count)
    return accepted


def main(args):
    lfw_root = Path(args.lfw_dir).expanduser()
    if not lfw_root.exists():
        raise FileNotFoundError(f"LFW root does not exist: {lfw_root}")

    out_root = Path(args.output_root).expanduser()
    identities_dir = out_root / "identities"
    landmarks_dir = out_root / "identity_landmarks"
    thatcher_dir = out_root / "thatcher"
    for d in (identities_dir, landmarks_dir, thatcher_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[init] MediaPipe FaceMesh refine_landmarks={args.refine}")
    print(f"[init] target_size={args.target_size}px, min_iod={args.min_iod_px}px")
    with mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=args.refine,
        min_detection_confidence=0.7,
    ) as mp_face:
        print(f"[select] sampling {args.target_n} identities from {lfw_root}, seed={args.seed}")
        accepted = select_identities(
            lfw_root, mp_face, args.target_n, args.target_size, args.seed, args.min_iod_px,
        )

    if len(accepted) < args.target_n:
        print(f"[warn] only {len(accepted)} identities passed QC (asked for {args.target_n})")
    if not accepted:
        raise RuntimeError(
            "No identities passed QC. Try lowering --min_iod_px or relaxing tilt threshold."
        )

    identity_records: list[Identity] = []
    manifest_rows: list[dict] = []
    print(f"[gen] generating Thatcher battery → {thatcher_dir}")
    for idx, s in enumerate(tqdm(accepted, desc="Thatcher")):
        identity_id = idx
        rgb = s["rgb"]
        lm_arr = s["lm_arr"]

        # Persist canonical identity crop + landmarks (for other paradigms)
        canonical_path = identities_dir / f"identity_{identity_id:04d}.png"
        Image.fromarray(rgb).save(canonical_path)
        lm_json_path = landmarks_dir / f"identity_{identity_id:04d}.json"
        with open(lm_json_path, "w") as f:
            json.dump({
                "identity_id": identity_id,
                "lfw_name": s["lfw_name"],
                "lfw_source_path": s["lfw_source_path"],
                "target_size": args.target_size,
                "landmarks": [[float(x), float(y)] for x, y in lm_arr.tolist()],
                "iod_px": s["iod_px"],
                "tilt_deg": s["tilt_deg"],
            }, f)
        identity_records.append(Identity(
            identity_id=identity_id,
            lfw_name=s["lfw_name"],
            lfw_source_path=s["lfw_source_path"],
            canonical_png_path=str(canonical_path),
            landmarks_json_path=str(lm_json_path),
        ))

        # Build 4 Thatcher versions
        v1 = rgb.copy()
        v2 = thatcherize(rgb, lm_arr,
                         eye_pad=args.eye_pad, mouth_pad=args.mouth_pad,
                         blur_sigma=args.blur_sigma)
        v3 = rotate_180(rgb)
        v4 = rotate_180(v2)
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
                "lfw_name": s["lfw_name"],
                "paradigm": "thatcher",
                "condition": vname,
                "orientation": orient,
                "thatcherized": int(thatch_flag),
                "path": str(path),
            })

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

    print(f"[done] {len(identity_records)} identities, {len(manifest_rows)} Thatcher stimuli")
    print(f"  identity manifest: {identity_manifest_path}")
    print(f"  thatcher manifest: {manifest_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lfw_dir", required=True, help="root of lfw_funneled (contains one subdir per identity)")
    p.add_argument("--output_root", required=True, help="output root for stimuli")
    p.add_argument("--target_n", type=int, default=200)
    p.add_argument("--target_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--eye_pad", type=int, default=8)
    p.add_argument("--mouth_pad", type=int, default=10)
    p.add_argument("--blur_sigma", type=float, default=3.0)
    p.add_argument("--min_iod_px", type=float, default=30.0,
                   help="reject identities with inter-ocular distance below this many pixels")
    p.add_argument("--refine", action="store_true", default=True,
                   help="use refine_landmarks=True (iris, finer eye landmarks)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
