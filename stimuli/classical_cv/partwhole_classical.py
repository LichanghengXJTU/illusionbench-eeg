"""stimuli/classical_cv/partwhole_classical.py — Tanaka 1997 part-whole via classical CV.

Per Tanaka & Sengco (1997):
  V1 whole_target = rgb_i
  V2 whole_foil   = rgb_i with j's eye region pasted in
  V3 part_target  = i's eye region on neutral 128-gray canvas
  V4 part_foil    = j's eye region on neutral 128-gray canvas (same position)

Pipeline:
  - dlib 68-pt landmarks (cached from Phase 2)
  - Convex hull of both eye+brow regions (combined polygon)
  - Rigid affine align j's eye region to i's eye position
  - Reinhard LAB color transfer: j's eye block → match i's skin around eyes
  - cv2.seamlessClone(NORMAL_CLONE) j's eye block into i for V2
  - V3/V4: paste eye blocks onto gray canvas with soft polygon mask
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stimuli.classical_cv.color_transfer import reinhard_transfer
from stimuli.classical_cv.composite_classical import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX, rigid_align, make_contact_sheet,
)
from stimuli.classical_cv.face_pipeline import (
    LEFT_EYE_REGION, RIGHT_EYE_REGION,
)


BOTH_EYES_REGION = LEFT_EYE_REGION + RIGHT_EYE_REGION


def eye_hull_and_mask(landmarks: np.ndarray, shape: tuple[int, int],
                       pad_frac: float = 0.06) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Build convex hull + binary mask for the combined eye+brow region.
    Returns (hull_points, mask, bbox). pad_frac expands the hull by % of face IPD."""
    h, w = shape
    pts = landmarks[BOTH_EYES_REGION].astype(np.float32)
    hull = cv2.convexHull(pts.astype(np.int32))
    le_c = landmarks[36:42].mean(axis=0)
    re_c = landmarks[42:48].mean(axis=0)
    iod = float(np.linalg.norm(le_c - re_c))
    # Expand hull outward by pad_frac × iod
    centroid = hull.reshape(-1, 2).mean(axis=0)
    expanded = []
    for p in hull.reshape(-1, 2):
        vec = p - centroid
        n = np.linalg.norm(vec)
        if n > 0:
            p_exp = p + (vec / n) * (iod * pad_frac)
        else:
            p_exp = p
        expanded.append(p_exp)
    hull_exp = np.array(expanded, dtype=np.int32).reshape(-1, 1, 2)
    # Clip to image bounds
    hull_exp[:, 0, 0] = np.clip(hull_exp[:, 0, 0], 0, w - 1)
    hull_exp[:, 0, 1] = np.clip(hull_exp[:, 0, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_exp, 255)
    xs = hull_exp[:, 0, 0]; ys = hull_exp[:, 0, 1]
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return hull_exp, mask, bbox


def annulus_skin_pixels(landmarks: np.ndarray, eye_mask: np.ndarray,
                          rgb: np.ndarray, ring_px: int = 30) -> np.ndarray:
    """Sample skin pixels just outside the eye hull (used as Reinhard reference)."""
    dilated = cv2.dilate(eye_mask, np.ones((ring_px, ring_px), np.uint8))
    ring = (dilated > 0) & (eye_mask == 0)
    if not ring.any():
        return rgb.reshape(-1, 3)[:1000]
    return rgb[ring]


def make_partwhole(rgb_i: np.ndarray, lm_i: np.ndarray,
                    rgb_j: np.ndarray, lm_j: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb_i.shape[:2]
    # Align j to i's frame
    j_aligned, lm_j_aligned = rigid_align(rgb_j, lm_j, lm_i, (h, w))
    # Build i's and j_aligned's eye masks (they should now be in similar positions)
    hull_i, mask_i, bbox_i = eye_hull_and_mask(lm_i, (h, w))
    hull_j, mask_j, bbox_j = eye_hull_and_mask(lm_j_aligned, (h, w))

    # For V2: paste j_aligned's eye region into i with Reinhard color transfer
    # First isolate j's eye block; then color-match to i's skin around eyes
    skin_pixels_i = annulus_skin_pixels(lm_i, mask_i, rgb_i)
    j_recolor = reinhard_transfer(j_aligned, skin_pixels_i[None, :, :])

    # Use mask_i as the editable region in i (where to put j's eyes)
    # The j_recolor image is full-frame; seamlessClone takes a center + mask
    cx, cy = (bbox_i[0] + bbox_i[2]) // 2, (bbox_i[1] + bbox_i[3]) // 2
    try:
        v2 = cv2.seamlessClone(j_recolor, rgb_i, mask_i,
                                 (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error as e:
        # alpha fallback
        alpha = cv2.GaussianBlur(mask_i, (0, 0), 4, 4).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        v2 = (rgb_i.astype(np.float32) * (1 - alpha) +
              j_recolor.astype(np.float32) * alpha).astype(np.uint8)

    # V3: i's eye region on neutral gray, with soft polygon mask
    gray = np.full_like(rgb_i, 128)
    alpha_soft = cv2.GaussianBlur(mask_i, (0, 0), 4, 4).astype(np.float32) / 255.0
    alpha_soft = alpha_soft[..., None]
    v3 = (gray.astype(np.float32) * (1 - alpha_soft) +
          rgb_i.astype(np.float32) * alpha_soft).astype(np.uint8)
    # V4: j's eye region (aligned, color-matched to i's surrounding) on gray
    v4 = (gray.astype(np.float32) * (1 - alpha_soft) +
          j_recolor.astype(np.float32) * alpha_soft).astype(np.uint8)
    v1 = rgb_i.copy()
    return v1, v2, v3, v4


def main(args):
    from stimuli.classical_cv.demographic_cluster import (
        face_descriptor, cluster_identities, within_cluster_pairs,
    )

    src_root = Path(args.phase2_dir)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    records = []
    for lm_path in sorted(src_root.glob("identity_*/landmarks.json")):
        meta = json.loads(lm_path.read_text())
        img_path = lm_path.parent / "thatcher_V1.png"
        rgb = np.array(Image.open(img_path).convert("RGB"))
        lm = np.array(meta["landmarks"], dtype=np.float32)
        bbox = tuple(meta["bbox"])
        records.append({
            "identity_id": meta["identity_id"], "ffhq_name": meta["ffhq_name"],
            "rgb": rgb, "lm": lm, "bbox": bbox,
            "descriptor": face_descriptor(rgb, lm, bbox),
        })
    n = len(records)
    print(f"[init] loaded {n} Phase-2 identities")
    cluster_map = cluster_identities(records, k=args.k_clusters)
    pairs = within_cluster_pairs(cluster_map)

    manifest = []; contact_sheets = []
    for r in records:
        i = r["identity_id"]; j = pairs.get(i, i)
        if j == i:
            print(f"  [skip] id {i} singleton"); continue
        rec_j = next(rr for rr in records if rr["identity_id"] == j)
        v1, v2, v3, v4 = make_partwhole(r["rgb"], r["lm"],
                                          rec_j["rgb"], rec_j["lm"])
        idir = out_root / f"identity_{i:04d}"
        idir.mkdir(parents=True, exist_ok=True)
        for cond, im in [("V1_whole_target", v1), ("V2_whole_foil", v2),
                          ("V3_part_target", v3), ("V4_part_foil", v4)]:
            p = idir / f"partwhole_{cond}.png"
            Image.fromarray(im).save(p)
            manifest.append({"identity_id": i, "paired_with": j,
                              "cluster": cluster_map[i],
                              "condition": cond, "path": str(p)})
        cs = idir / "contact.png"
        make_contact_sheet([v1, v2, v3, v4],
                           ["V1 whole_target", "V2 whole_foil",
                            "V3 part_target", "V4 part_foil"], cs)
        contact_sheets.append(cs)
        print(f"  [pair] id {i} ↔ id {j}")

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    if len(contact_sheets) >= 4:
        rows = [np.array(Image.open(p).convert("RGB"))
                for p in contact_sheets[:4]]
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        Image.fromarray(np.concatenate(padded, axis=0)).save(
            out_root / "qc_4identities.png")
        print(f"  combined QC: {out_root}/qc_4identities.png")

    print(f"\n[done] {len(manifest)//4} part-whole identities × 4 conditions")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase2_dir", default="data/classical_cv_phase2_thatcher")
    p.add_argument("--output_dir", default="data/classical_cv_phase3_partwhole")
    p.add_argument("--k_clusters", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
