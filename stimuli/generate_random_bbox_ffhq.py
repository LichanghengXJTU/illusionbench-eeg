"""stimuli/generate_random_bbox_ffhq.py — Q001 control: random-bbox FFHQ rotations.

DESIGN GOAL: discriminate between two hypotheses about why CLIP-class priors give
ISI ≈ 4-7 on FFHQ Thatcher (E002):
  H_face-specific: CLIP detects face-configural disruption (rotated eyes/brows/mouth)
  H_general-bias: CLIP's embedding has a general upright-orientation amplifier that
                  responds to ANY local perturbation more in upright than inverted

Cleanest control isolates ONE variable from E002: the SEMANTIC IDENTITY of the
rotated region. Same FFHQ identities, same algorithm, same total perturbation
area, but bbox LOCATIONS are random non-feature regions (cheek, forehead, neck,
background) instead of eyes/brows/mouth.

Outputs use the SAME naming convention as E002 (V1_upright_normal,
V2_upright_thatched, V3_inverted_normal, V4_inverted_thatched) so downstream
extract_embeddings and compute_metrics scripts work unchanged.

Logic:
  1. Read identity_landmarks JSONs from data/stimuli_ffhq (200 already-selected identities)
  2. For each identity, load original 1024×1024 PNG
  3. Compute exclusion zone: union of all FACEMESH_TESSELATION landmark positions, dilated
  4. Sample 3 random bboxes of sizes matched to the eye/eye/mouth bbox sizes used in
     E002, with centers outside the exclusion zone, non-overlapping
  5. Apply 180° rotation per bbox + soft alpha blend (reuse Thatcherize utilities)
  6. Save V1/V2/V3/V4 PNGs + manifest with paradigm="random_bbox" but same condition strings
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stimuli.generate_thatcher import (
    paste_with_alpha,
    rotate_180,
    LEFT_EYE_REGION,
    RIGHT_EYE_REGION,
    LIPS_PTS,
)


def get_bbox_from_landmarks(landmark_arr: np.ndarray, indices: list[int],
                            img_w: int, img_h: int, padding: int) -> tuple[int, int, int, int]:
    pts = landmark_arr[list(indices)]
    x1 = max(0, int(pts[:, 0].min() - padding))
    y1 = max(0, int(pts[:, 1].min() - padding))
    x2 = min(img_w, int(pts[:, 0].max() + padding))
    y2 = min(img_h, int(pts[:, 1].max() + padding))
    return x1, y1, x2, y2


def build_exclusion_mask(landmark_arr: np.ndarray, w: int, h: int, dilate_px: int) -> np.ndarray:
    """Boolean (h,w) mask: True = forbidden (covers face features)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # All 468/478 landmark pixels marked, then dilated to form a face envelope
    pts = landmark_arr.astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    for x, y in pts:
        mask[y, x] = 1
    kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def sample_random_bbox(w: int, h: int, bw: int, bh: int,
                       exclude_mask: np.ndarray, occupied: list[tuple[int, int, int, int]],
                       rng: np.random.Generator, max_tries: int = 200) -> tuple[int, int, int, int] | None:
    """Try to sample a (x1,y1,x2,y2) bbox of size (bw,bh) whose centre lies OUTSIDE
    exclude_mask and does not overlap any of `occupied` boxes. Returns None on failure."""
    for _ in range(max_tries):
        x1 = int(rng.integers(0, max(1, w - bw)))
        y1 = int(rng.integers(0, max(1, h - bh)))
        x2 = x1 + bw
        y2 = y1 + bh
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if exclude_mask[cy, cx]:
            continue
        # Also require the whole bbox region to be < 30% covered by exclude_mask
        coverage = exclude_mask[y1:y2, x1:x2].mean()
        if coverage > 0.30:
            continue
        # Reject overlap with already-placed boxes
        bad = False
        for (ox1, oy1, ox2, oy2) in occupied:
            if not (x2 <= ox1 or x1 >= ox2 or y2 <= oy1 or y1 >= oy2):
                bad = True
                break
        if bad:
            continue
        return (x1, y1, x2, y2)
    return None


def rotate_180_in_bboxes(img_rgb: np.ndarray, bboxes: list[tuple[int, int, int, int]],
                         blur_sigma: float) -> np.ndarray:
    result = img_rgb.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        patch = result[y1:y2, x1:x2].copy()
        patch_rot = cv2.rotate(patch, cv2.ROTATE_180)
        paste_with_alpha(result, patch_rot, bbox, blur_sigma)
    return result


@dataclass
class Identity:
    identity_id: int
    ffhq_name: str
    canonical_png_path: str
    landmarks_json_path: str
    bboxes_used: str   # JSON-encoded list of (x1,y1,x2,y2)


def main(args):
    src_root = Path(args.src_root)
    src_identities = src_root / "identities"
    src_landmarks = src_root / "identity_landmarks"
    src_identity_manifest = src_root / "identity_manifest.csv"

    out_root = Path(args.output_root)
    stim_dir = out_root / "random_bbox"
    identities_dir = out_root / "identities"   # symlink references
    landmarks_dir = out_root / "identity_landmarks"
    stim_dir.mkdir(parents=True, exist_ok=True)
    identities_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    # Read existing FFHQ identity manifest
    src_records = []
    with open(src_identity_manifest) as f:
        for row in csv.DictReader(f):
            src_records.append(row)
    print(f"[init] read {len(src_records)} source identities from {src_identity_manifest}")

    rng = np.random.default_rng(args.seed)
    manifest_rows = []
    identity_records = []

    for src in tqdm(src_records, desc="random_bbox"):
        identity_id = int(src["identity_id"])
        png_path = Path(src["canonical_png_path"])
        lm_path = Path(src["landmarks_json_path"])
        if not png_path.exists() or not lm_path.exists():
            print(f"  [skip] id{identity_id}: missing files")
            continue
        # Load image
        bgr = cv2.imread(str(png_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Load cached landmarks
        with open(lm_path) as f:
            lm_data = json.load(f)
        lm_arr = np.array(lm_data["landmarks"], dtype=np.float32)
        # Match bbox sizes to the original Thatcher eye / eye / mouth bboxes (E002 used
        # eye_pad=20, mouth_pad=25 at 1024 native, scaled with image size).
        eye_pad = lm_data.get("eye_pad_used", max(8, int(min(h, w) * 0.02)))
        mouth_pad = lm_data.get("mouth_pad_used", max(10, int(min(h, w) * 0.025)))
        blur_sigma = lm_data.get("blur_sigma_used", max(3.0, min(h, w) * 0.005))

        left_eye_bbox = get_bbox_from_landmarks(lm_arr, LEFT_EYE_REGION, w, h, eye_pad)
        right_eye_bbox = get_bbox_from_landmarks(lm_arr, RIGHT_EYE_REGION, w, h, eye_pad)
        mouth_bbox = get_bbox_from_landmarks(lm_arr, LIPS_PTS, w, h, mouth_pad)
        feature_bbox_sizes = [
            (left_eye_bbox[2] - left_eye_bbox[0], left_eye_bbox[3] - left_eye_bbox[1]),
            (right_eye_bbox[2] - right_eye_bbox[0], right_eye_bbox[3] - right_eye_bbox[1]),
            (mouth_bbox[2] - mouth_bbox[0], mouth_bbox[3] - mouth_bbox[1]),
        ]

        # Exclusion mask: dilated face landmarks
        excl = build_exclusion_mask(lm_arr, w, h, dilate_px=args.exclusion_dilate_px)

        # Sample 3 random bboxes matching feature sizes
        random_bboxes = []
        for (bw, bh) in feature_bbox_sizes:
            bbox = sample_random_bbox(w, h, bw, bh, excl, random_bboxes, rng,
                                      max_tries=args.max_tries)
            if bbox is not None:
                random_bboxes.append(bbox)
        if len(random_bboxes) < 3:
            # Fallback: relax exclusion (only background of image, not face oval)
            print(f"  [warn] id{identity_id}: only {len(random_bboxes)} bboxes placed, skipping")
            continue

        # Build 4 versions
        v1 = rgb.copy()
        v2 = rotate_180_in_bboxes(rgb, random_bboxes, blur_sigma)
        v3 = rotate_180(rgb)
        v4 = rotate_180(v2)

        # Symlink identity png + landmark json into out_root for self-contained manifest
        ident_sym = identities_dir / f"identity_{identity_id:04d}.png"
        lm_sym = landmarks_dir / f"identity_{identity_id:04d}.json"
        if not ident_sym.exists():
            try:
                ident_sym.symlink_to(png_path.resolve())
            except FileExistsError:
                pass
        if not lm_sym.exists():
            try:
                lm_sym.symlink_to(lm_path.resolve())
            except FileExistsError:
                pass

        identity_records.append(Identity(
            identity_id=identity_id,
            ffhq_name=src.get("lfw_name", src.get("ffhq_name", "")),
            canonical_png_path=str(ident_sym),
            landmarks_json_path=str(lm_sym),
            bboxes_used=json.dumps([list(map(int, b)) for b in random_bboxes]),
        ))

        for vname, vimg, orient, thatch_flag in [
            ("V1_upright_normal", v1, "upright", False),
            ("V2_upright_thatched", v2, "upright", True),
            ("V3_inverted_normal", v3, "inverted", False),
            ("V4_inverted_thatched", v4, "inverted", True),
        ]:
            path = stim_dir / f"identity_{identity_id:04d}_{vname}.png"
            Image.fromarray(vimg).save(path)
            manifest_rows.append({
                "stim_id": f"thatcher_id{identity_id:04d}_{vname}",
                "identity_id": identity_id,
                "lfw_name": src.get("lfw_name", src.get("ffhq_name", "")),
                "paradigm": "random_bbox",
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

    print(f"[done] {len(identity_records)} identities, {len(manifest_rows)} stimuli")
    print(f"  identity manifest: {identity_manifest_path}")
    print(f"  stimulus manifest: {manifest_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True,
                   help="directory containing FFHQ identities + landmarks (e.g. data/stimuli_ffhq)")
    p.add_argument("--output_root", required=True)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--exclusion_dilate_px", type=int, default=80,
                   help="how far to dilate face landmark mask to forbid bbox centres")
    p.add_argument("--max_tries", type=int, default=400)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
