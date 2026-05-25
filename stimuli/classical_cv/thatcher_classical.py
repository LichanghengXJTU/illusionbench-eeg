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
)
from stimuli.classical_cv.precise_polygons import (
    eye_brow_polygon, eye_only_polygon, mouth_polygon,
    polygon_bbox, polygon_centroid,
    LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW,
)


def thatcherize_classical(img_rgb: np.ndarray, landmarks: np.ndarray,
                            pad: int = 12) -> np.ndarray:
    """Thatcher with **centroid-anchored** 180° rotation per region.

    Per-region procedure (per user feedback 2026-05-25):
      1. Build precise polygon with proper pads (brow_up, eye-bag, lateral
         makeup zone — see precise_polygons.eye_brow_polygon).
      2. Compute polygon CENTROID (cx, cy).
      3. Build centroid-symmetric bbox = (cx ± dx_max, cy ± dy_max) so
         that rotating the bbox patch 180° rotates the FEATURE around its
         own centroid (not around bbox edge = lower-lip / upper-brow).
      4. Build mask M from polygon (in bbox-local coords).
         Build M_rot = M rotated 180° (= same mask reflected around bbox center).
         Use mask_union = M ∪ M_rot for seamlessClone target region — covers
         BOTH the original feature position AND the rotated position, so any
         residue of the original (in the wedge where rotated doesn't reach)
         still gets replaced by the rotated patch content.
      5. cv2.seamlessClone(NORMAL_CLONE) — Poisson solves the Laplace at
         mask boundary, hiding the seam in surrounding skin gradient.
    """
    h, w = img_rgb.shape[:2]
    le_c = landmarks[LEFT_EYE].mean(axis=0)
    re_c = landmarks[RIGHT_EYE].mean(axis=0)
    iod = float(np.linalg.norm(le_c - re_c))

    # Thompson 1980 canonical: rotate EYE (not brow) + MOUTH only.
    # Brow stays put — eliminates the "brows clearly visible after flip"
    # problem entirely (no flip → no visibility issue).
    polys = [
        ("L_eye", eye_only_polygon(landmarks, LEFT_EYE,  iod=iod)),
        ("R_eye", eye_only_polygon(landmarks, RIGHT_EYE, iod=iod)),
        ("mouth", mouth_polygon(landmarks, iod=iod)),
    ]
    out = img_rgb.copy()
    for name, poly in polys:
        cx, cy = polygon_centroid(poly)
        # Centroid-symmetric bbox: ensures rotation around centroid
        xs = poly[:, 0, 0]; ys = poly[:, 0, 1]
        dx_max = max(int(abs(xs.max() - cx)), int(abs(cx - xs.min()))) + pad
        dy_max = max(int(abs(ys.max() - cy)), int(abs(cy - ys.min()))) + pad
        # Clip to image bounds while keeping symmetry around centroid
        max_dx = min(dx_max, cx, w - 1 - cx)
        max_dy = min(dy_max, cy, h - 1 - cy)
        x1, x2 = cx - max_dx, cx + max_dx
        y1, y2 = cy - max_dy, cy + max_dy
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        # Extract bbox patch + rotate 180° around bbox center (= centroid)
        patch = out[y1:y2, x1:x2].copy()
        patch_rot = cv2.rotate(patch, cv2.ROTATE_180)
        # Build polygon mask in bbox-local coords
        poly_local = poly.copy()
        poly_local[:, 0, 0] -= x1
        poly_local[:, 0, 1] -= y1
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_local], 255)
        # Union with rotated mask → covers any uncovered ghost wedge
        mask_rot = cv2.rotate(mask, cv2.ROTATE_180)
        mask_union = cv2.bitwise_or(mask, mask_rot)
        # seamlessClone — NORMAL_CLONE preserves rotated content,
        # smooths boundary into surrounding skin via Poisson editing.
        center = (cx, cy)
        try:
            out = cv2.seamlessClone(patch_rot, out, mask_union,
                                      center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            alpha = (mask_union.astype(np.float32) / 255.0)[..., None]
            base = out[y1:y2, x1:x2].astype(np.float32)
            new = patch_rot.astype(np.float32)
            out[y1:y2, x1:x2] = np.clip(base * (1 - alpha) + new * alpha,
                                          0, 255).astype(np.uint8)
            print(f"  [warn] seamlessClone failed on {name}, alpha fallback: {e}")
    return out


def overlay_polygons(img_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw the precise eye/brow/mouth polygons on top of the source for QC."""
    h, w = img_rgb.shape[:2]
    le_c = landmarks[LEFT_EYE].mean(axis=0)
    re_c = landmarks[RIGHT_EYE].mean(axis=0)
    iod = float(np.linalg.norm(le_c - re_c))
    polys = [
        ("L_eye", eye_only_polygon(landmarks, LEFT_EYE,  iod=iod), (0, 255, 0)),
        ("R_eye", eye_only_polygon(landmarks, RIGHT_EYE, iod=iod), (0, 255, 0)),
        ("mouth", mouth_polygon(landmarks, iod=iod), (255, 0, 0)),
    ]
    out = img_rgb.copy()
    for name, poly, color in polys:
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=3)
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
        v0_overlay = overlay_polygons(rgb, rec.landmarks)

        Image.fromarray(v0_overlay).save(idir / "polygon_overlay.png")

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
