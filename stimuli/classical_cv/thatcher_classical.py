"""stimuli/classical_cv/thatcher_classical.py — Thatcher via Poisson cloning.

Pure classical CV implementation of Thompson 1980's Thatcher illusion.

For each filtered FFHQ source:
  1. dlib 68-pt landmarks (from face_pipeline.py)
  2. Build convex-hull polygons for: LEFT eye+brow, RIGHT eye+brow, MOUTH
  3. For each polygon:
     a. Extract enclosing bbox patch from V1
     b. Rotate the bbox patch 180°
     c. Build a binary mask of the convex hull within the bbox
     d. cv2.seamlessClone(NORMAL_CLONE) the rotated patch back at the hull
        centroid → Poisson editing makes the boundary invisible by solving
        Laplace equation that matches gradients at the seam
  4. Save V1 (original), V2 (thatched), V3 (rot180 V1), V4 (rot180 V2)

Pixel-baseline ISI = 1.000 by construction (V3 = rot180(V1), V4 = rot180(V2)).

Reference:
  Pérez, Gangnet & Blake (2003) "Poisson Image Editing", SIGGRAPH.
  Thompson (1980) "Margaret Thatcher: a new illusion."
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stimuli.classical_cv.face_pipeline import (
    FacePipeline, default_filter, iter_ffhq_tar,
    LEFT_EYE_REGION, RIGHT_EYE_REGION, MOUTH_REGION,
)


def hull_and_bbox(landmarks: np.ndarray, idx: list[int], img_h: int, img_w: int,
                    pad: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (hull_points_in_image_coords, padded_bbox)."""
    pts = landmarks[idx].astype(np.int32)
    hull = cv2.convexHull(pts)
    x1 = max(0, int(pts[:, 0].min()) - pad)
    y1 = max(0, int(pts[:, 1].min()) - pad)
    x2 = min(img_w, int(pts[:, 0].max()) + pad)
    y2 = min(img_h, int(pts[:, 1].max()) + pad)
    return hull, (x1, y1, x2, y2)


def thatcherize_classical(img_rgb: np.ndarray, landmarks: np.ndarray,
                            pad: int = 8) -> np.ndarray:
    """Apply Thatcher transform with Poisson seamless cloning.

    Rotates left eye+brow, right eye+brow, and mouth regions 180° in place,
    using convex-hull masks and cv2.seamlessClone for invisible blending."""
    h, w = img_rgb.shape[:2]
    out = img_rgb.copy()
    # Process each region. Mouth padded slightly larger because it has more
    # surrounding texture to blend (lips → chin/cheek skin transition).
    regions = [
        ("L_eye", LEFT_EYE_REGION,  pad),
        ("R_eye", RIGHT_EYE_REGION, pad),
        ("mouth", MOUTH_REGION,     pad + 4),
    ]
    for name, idx, region_pad in regions:
        hull, bbox = hull_and_bbox(landmarks, idx, h, w, region_pad)
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue

        # 1. Extract bbox patch from CURRENT working image (out),
        #    so successive region edits don't clobber each other.
        patch = out[y1:y2, x1:x2].copy()

        # 2. Rotate the patch 180°
        patch_rot = cv2.rotate(patch, cv2.ROTATE_180)

        # 3. Build the hull mask in BBOX-LOCAL coordinates, and also the
        #    rotated version (since the patch is now rotated 180°, the mask
        #    that selects "feature interior" must also be rotated).
        hull_local = hull.copy()
        hull_local[:, 0, 0] -= x1
        hull_local[:, 0, 1] -= y1
        mask_local = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillConvexPoly(mask_local, hull_local, 255)
        mask_local_rot = cv2.rotate(mask_local, cv2.ROTATE_180)
        # Use the rotated mask: it tells seamlessClone which pixels of
        # patch_rot correspond to "feature content."
        mask_for_clone = mask_local_rot

        # 4. seamlessClone center = the patch center in destination coords
        #    (i.e., the hull's bounding-box center in the full image).
        center = (x1 + bw // 2, y1 + bh // 2)

        # 5. Poisson seamless clone (NORMAL_CLONE preserves the source patch's
        #    content while smoothing the boundary to match destination gradients).
        try:
            out = cv2.seamlessClone(patch_rot, out, mask_for_clone,
                                      center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            # Fallback: simple alpha blend with the mask (if seamlessClone
            # fails on edge cases — e.g., mask near image border)
            alpha = (mask_for_clone.astype(np.float32) / 255.0)[..., None]
            base = out[y1:y2, x1:x2].astype(np.float32)
            new = patch_rot.astype(np.float32)
            blend = base * (1 - alpha) + new * alpha
            out[y1:y2, x1:x2] = np.clip(blend, 0, 255).astype(np.uint8)
            print(f"  [warn] seamlessClone failed on {name}, used alpha "
                  f"fallback: {e}")
    return out


def make_contact_sheet(images, titles, out_path, tile=384):
    n = len(images)
    strip = 30
    sheet = np.full((tile + strip, tile * n, 3), 255, dtype=np.uint8)
    for i, (im, t) in enumerate(zip(images, titles)):
        sheet[:tile, i*tile:(i+1)*tile] = cv2.resize(
            im, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, t, (i*tile + 10, tile + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(sheet).save(out_path)


def main(args):
    pipe = FacePipeline(args.predictor)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    n_target = args.target_n

    manifest = []
    n_scanned = 0; n_passed = 0
    contact_sheets = []
    for src_name, rgb in iter_ffhq_tar(Path(args.tar), max_n=args.scan_cap,
                                         shuffle=True, seed=args.seed):
        n_scanned += 1
        if n_passed >= n_target:
            break
        faces = pipe.detect_and_landmark(rgb)
        if not faces:
            continue
        rec = max(faces, key=lambda r: (r.bbox[2] - r.bbox[0]) *
                                         (r.bbox[3] - r.bbox[1]))
        rec.src_name = src_name
        ok, _ = default_filter(rec)
        if not ok:
            continue

        idx = n_passed
        idir = out_root / f"identity_{idx:04d}"
        idir.mkdir(parents=True, exist_ok=True)

        v1 = rgb.copy()
        v2 = thatcherize_classical(rgb, rec.landmarks, pad=args.pad)
        v3 = cv2.rotate(v1, cv2.ROTATE_180)
        v4 = cv2.rotate(v2, cv2.ROTATE_180)

        for cond, im in [("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)]:
            p = idir / f"thatcher_{cond}.png"
            Image.fromarray(im).save(p)
            manifest.append({"identity_id": idx, "ffhq_name": src_name,
                              "condition": cond, "path": str(p)})

        # Per-identity contact sheet
        cs = idir / "contact.png"
        make_contact_sheet([v1, v2, v3, v4],
                           ["V1 upright_normal", "V2 upright_thatched",
                            "V3 inverted_normal", "V4 inverted_thatched"], cs)
        contact_sheets.append(cs)

        # Also save landmarks for downstream use (composite/PW)
        lm_path = idir / "landmarks.json"
        lm_path.write_text(json.dumps({
            "identity_id": idx, "ffhq_name": src_name,
            "landmarks": rec.landmarks.tolist(),
            "iod_px": rec.iod_px, "tilt_deg": rec.tilt_deg,
            "yaw_score": rec.yaw_score, "bbox": list(rec.bbox),
        }, indent=2))

        n_passed += 1
        print(f"  [{n_passed}/{n_target}] {src_name} → identity_{idx:04d}")

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    # Combined QC: first 4 identities, stacked vertically
    if len(contact_sheets) >= 4:
        rows = [np.array(Image.open(p).convert("RGB"))
                for p in contact_sheets[:4]]
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255,
                               dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        Image.fromarray(np.concatenate(padded, axis=0)).save(
            out_root / "qc_4identities.png")
        print(f"  combined QC: {out_root}/qc_4identities.png")

    print(f"\n[done] scanned {n_scanned}, generated {n_passed} identities × "
          f"4 conditions = {len(manifest)} stimuli")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor",
                   default="/workspace/models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--tar",
                   default="/workspace/.hf_cache/hub/datasets--gaunernst--ffhq-1024-wds/snapshots/d74f1f1f59e3bbe975bee29872b9bef827314577/00000.tar")
    p.add_argument("--output_dir", default="data/classical_cv_phase2_thatcher")
    p.add_argument("--target_n", type=int, default=20)
    p.add_argument("--scan_cap", type=int, default=200)
    p.add_argument("--pad", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260525)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
