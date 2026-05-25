"""stimuli/classical_cv/partwhole_classical.py — Tanaka 1997 part-whole,
three-feature version (eyes / nose / mouth).

Per Tanaka & Sengco (1997): each feature gets its own 2-AFC test under
WHOLE-face vs PART-isolated conditions. Per-feature conditions:

  V1 whole_target = i_original
  V2 whole_foil   = i with donor_f's feature swapped in
  V3 part_target  = i's feature on a gray=128 canvas at same bbox
  V4 part_foil    = donor_f's (aligned + recolored) feature on the SAME
                    gray=128 canvas at the SAME bbox

Pipeline per feature:
  1. Pick a donor j (per donor_match.py)
  2. Rigid affine align j → i using eye centers (similarity transform)
  3. NOSE only: secondary similarity using bridge endpoints (lm 27→30)
     so nose lengths match (eye-center align alone doesn't normalize this)
  4. Build i's feature polygon (precise_polygons.*)
  5. Reinhard LAB color transfer: j_aligned → match i's skin annulus
     around the feature polygon
  6. cv2.seamlessClone(NORMAL_CLONE) j's feature → i's frame for V2
  7. V3/V4: soft-alpha composite onto gray=128 canvas with same bbox

Post-hoc: SSIM(V1, V2) inside the feature bbox; if > 0.98 → reject pair,
re-sample donor.

Local-dev usage (4-ID QC):
  python -m stimuli.classical_cv.partwhole_classical \
    --source_dir data/local_qc_partwhole \
    --output_dir data/local_partwhole_v1 \
    --predictor models/shape_predictor_68_face_landmarks.dat \
    --recognizer models/dlib_face_recognition_resnet_model_v1.dat \
    --n_qc 4

Scale usage (server, all 26k):
  see partwhole_parallel.py (forthcoming)
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from stimuli.classical_cv.color_transfer import reinhard_transfer
from stimuli.classical_cv.composite_classical import (
    rigid_align, LEFT_EYE_IDX, RIGHT_EYE_IDX,
)
from stimuli.classical_cv.precise_polygons import (
    both_eyes_polygon, nose_polygon, mouth_polygon,
    both_eyes_tight_polygon, nose_tight_polygon, mouth_tight_polygon,
    polygon_centroid, polygon_bbox, polygon_mask,
)
from stimuli.classical_cv.face_embedding import FaceEmbedder
from stimuli.classical_cv.donor_match import (
    build_attr, candidate_pool, IdAttr,
)


# === per-feature polygon dispatch ============================================

def _build_polygon(feature: str, lm: np.ndarray, iod: float) -> np.ndarray:
    """PADDED polygon used by V2 swap (Poisson needs surrounding skin)."""
    if feature == "eye":
        return both_eyes_polygon(lm, iod=iod)
    if feature == "nose":
        return nose_polygon(lm, iod=iod)
    if feature == "mouth":
        return mouth_polygon(lm, iod=iod)
    raise ValueError(f"unknown feature {feature}")


def _build_tight_polygon(feature: str, lm: np.ndarray, iod: float) -> np.ndarray:
    """TIGHT polygon for V3/V4 part isolation — feature-only, no padding.
    On a gray canvas this leaves only the actual feature visible (Tanaka-style),
    not the surrounding skin (which blends invisibly with the gray)."""
    if feature == "eye":
        return both_eyes_tight_polygon(lm)
    if feature == "nose":
        return nose_tight_polygon(lm, iod=iod)
    if feature == "mouth":
        return mouth_tight_polygon(lm)
    raise ValueError(f"unknown feature {feature}")


# === nose-specific second alignment (bridge length match) ===================

def _renose_align(j_aligned: np.ndarray, lm_j_aligned: np.ndarray,
                  lm_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Secondary similarity transform anchored on nose-bridge endpoints
    (lm 27, lm 30). Normalizes j's nose length/orientation to match i's.
    Applied globally — eyes get slightly mis-aligned but we only USE the
    nose region of the output, and skin tone is matched by Reinhard."""
    src = np.float32([lm_j_aligned[27], lm_j_aligned[30]])
    dst = np.float32([lm_i[27], lm_i[30]])
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:
        return j_aligned, lm_j_aligned
    h, w = j_aligned.shape[:2]
    warped = cv2.warpAffine(j_aligned, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    lm_h = np.concatenate([lm_j_aligned, np.ones((68, 1))], axis=1)
    lm_warped = (M @ lm_h.T).T
    return warped, lm_warped.astype(np.float32)


# === skin annulus around feature polygon (Reinhard reference) ===============

def _annulus_skin_pixels(rgb: np.ndarray, mask: np.ndarray,
                         ring_px: int = 30) -> np.ndarray:
    """Pixels just outside the feature polygon (a ring of skin)."""
    dilated = cv2.dilate(mask, np.ones((ring_px, ring_px), np.uint8))
    ring = (dilated > 0) & (mask == 0)
    if not ring.any():
        return rgb.reshape(-1, 3)[:1000]
    return rgb[ring]


# === single-feature swap ====================================================

def swap_one_feature(feature: str,
                     rgb_i: np.ndarray, lm_i: np.ndarray,
                     rgb_j: np.ndarray, lm_j: np.ndarray,
                     iod_i: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (V1, V2, V3, V4, meta_dict)."""
    h, w = rgb_i.shape[:2]
    # 1. Rigid align j → i (eye centers)
    j_aligned, lm_j_aligned = rigid_align(rgb_j, lm_j, lm_i, (h, w))
    # 2. Nose-specific second alignment
    if feature == "nose":
        j_aligned, lm_j_aligned = _renose_align(j_aligned, lm_j_aligned, lm_i)
    # 3. Build feature polygons.
    #    For V2 swap: take UNION of i's polygon AND j's polygon (in i's frame).
    #    This is the ghost-union trick from Thatcher V5 — guarantees i's
    #    original feature is fully covered by j's content (no ghost of
    #    original nose / eye / lip remaining).
    poly_pad_i = _build_polygon(feature, lm_i, iod_i)
    mask_pad_i = polygon_mask(poly_pad_i, (h, w))
    poly_pad_j = _build_polygon(feature, lm_j_aligned, iod_i)
    mask_pad_j = polygon_mask(poly_pad_j, (h, w))
    mask_union = cv2.bitwise_or(mask_pad_i, mask_pad_j)
    M = cv2.moments(mask_union)
    if M["m00"] > 0:
        cx_u = int(M["m10"] / M["m00"]); cy_u = int(M["m01"] / M["m00"])
    else:
        cx_u, cy_u = polygon_centroid(poly_pad_i)
    bbox_pad = polygon_bbox(poly_pad_i, (h, w), pad=4)  # for SSIM crop

    poly_tight = _build_tight_polygon(feature, lm_i, iod_i)
    mask_tight = polygon_mask(poly_tight, (h, w))
    bbox_tight = polygon_bbox(poly_tight, (h, w), pad=2)

    # 4. Reinhard: recolor j_aligned to match i's skin annulus around UNION
    skin_ref = _annulus_skin_pixels(rgb_i, mask_union, ring_px=30)
    j_recolor = reinhard_transfer(j_aligned, skin_ref[None, :, :])
    # 5. V2 — seamlessClone j_recolor → i using mask_union (ghost-eliminating)
    try:
        v2 = cv2.seamlessClone(j_recolor, rgb_i, mask_union, (cx_u, cy_u),
                               cv2.NORMAL_CLONE)
    except cv2.error:
        alpha = cv2.GaussianBlur(mask_union, (0, 0), 5, 5).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        v2 = (rgb_i.astype(np.float32) * (1 - alpha) +
              j_recolor.astype(np.float32) * alpha).astype(np.uint8)
    # 6. V3/V4 — soft-alpha onto gray 128 canvas using TIGHT mask
    #    Only the feature itself (not surrounding skin) is visible on gray.
    gray = np.full_like(rgb_i, 128)
    alpha_soft = cv2.GaussianBlur(mask_tight, (0, 0), 2, 2).astype(np.float32) / 255.0
    alpha_soft = alpha_soft[..., None]
    v3 = (gray.astype(np.float32) * (1 - alpha_soft) +
          rgb_i.astype(np.float32) * alpha_soft).astype(np.uint8)
    v4 = (gray.astype(np.float32) * (1 - alpha_soft) +
          j_recolor.astype(np.float32) * alpha_soft).astype(np.uint8)
    v1 = rgb_i.copy()
    # 7. SSIM in padded bbox (where V2's swap is visible)
    x1, y1, x2, y2 = bbox_pad
    v1_crop = v1[y1:y2, x1:x2]; v2_crop = v2[y1:y2, x1:x2]
    s = float(ssim(v1_crop, v2_crop, channel_axis=2,
                   data_range=255)) if v1_crop.size > 0 else 1.0
    meta = {"ssim_v1v2_bbox": s,
            "bbox_pad": bbox_pad, "bbox_tight": bbox_tight}
    return v1, v2, v3, v4, meta


# === full 3-feature pipeline per identity ===================================

def _get_pool_with_relaxation(i: IdAttr, all_attrs: list[IdAttr]) -> list[IdAttr]:
    """Get candidate pool, relaxing constraints if empty (up to 3 steps).
    Relaxation order: only embedding window + skin LAB + face_ratio.
    Age-related constraints (iod_ratio_max, smoothness) STAY STRICT
    to avoid cross-age pairings even when the pool is small."""
    # Age constraints (iod_ratio, smooth_log) STAY STRICT across relaxation.
    # If a child cannot find an age-matched donor, the identity gets skipped
    # rather than producing a cross-age (broken) stimulus.
    starts = [
        # (d_low, d_high, lab_L, face_ratio, iod_ratio, smooth_log)
        (0.55, 0.85, 8.0,  0.10, 0.05, 0.30),
        (0.50, 0.90, 11.0, 0.15, 0.05, 0.30),
        (0.45, 0.95, 14.0, 0.20, 0.05, 0.30),
    ]
    for d_low, d_high, lab_L, fr, iod_r, sm in starts:
        pool = candidate_pool(i, all_attrs, d_low, d_high, lab_L, fr, iod_r, sm)
        if pool:
            return pool
    return []


def partwhole_one_identity(i: IdAttr, all_attrs: list[IdAttr],
                           rng: random.Random,
                           ssim_reject_above: float = 0.98,
                           max_donor_retries: int = 5
                           ) -> dict | None:
    """Run all three feature swaps for one identity i.

    Donor sampling: pick 3 DISTINCT donors initially from the candidate
    pool (one per feature). For any feature whose SSIM(V1,V2) > threshold
    (foil too similar to target), re-sample a new donor for THAT feature
    only (without replacement, from remaining pool).

    Returns dict {feature -> {V1..V4, donor_name, meta}} or None if no
    candidate pool exists at all."""
    iod_i = float(np.linalg.norm(
        i.lm[36:42].mean(0) - i.lm[42:48].mean(0)))
    pool = _get_pool_with_relaxation(i, all_attrs)
    if not pool:
        return None
    # Initial donor allocation: 3 distinct from pool if possible
    pool_shuffled = pool.copy(); rng.shuffle(pool_shuffled)
    feature_order = ["eye", "nose", "mouth"]
    initial = {}
    for k, feat in enumerate(feature_order):
        initial[feat] = pool_shuffled[k % len(pool_shuffled)]
    used = {initial[f].name for f in feature_order}

    out = {}
    for feature in feature_order:
        j = initial[feature]
        tried_names = {j.name}
        accepted = False
        attempts = 0
        last = None
        while not accepted and attempts < max_donor_retries:
            v1, v2, v3, v4, meta = swap_one_feature(
                feature, i.rgb, i.lm, j.rgb, j.lm, iod_i)
            last = (v1, v2, v3, v4, meta, j)
            if meta["ssim_v1v2_bbox"] <= ssim_reject_above:
                accepted = True
                break
            # SSIM too high — re-sample a different donor from pool
            remaining = [p for p in pool_shuffled if p.name not in tried_names]
            if not remaining:
                break
            j = rng.choice(remaining)
            tried_names.add(j.name)
            attempts += 1
        v1, v2, v3, v4, meta, j = last
        out[feature] = {
            "V1": v1, "V2": v2, "V3": v3, "V4": v4,
            "donor_name": j.name + ("" if accepted else "_BESTEFFORT"),
            "meta": meta,
        }
    return out


# === QC contact sheet =======================================================

def make_qc_sheet(identity_results: list[tuple[str, dict]],
                  out_path: Path, tile_size: int = 220) -> None:
    """Build a contact sheet: rows = identities, cols = [V1, V2_eye, V3_eye,
    V4_eye, V2_nose, V3_nose, V4_nose, V2_mouth, V3_mouth, V4_mouth].
    Columns are labeled at the top."""
    feature_order = ["eye", "nose", "mouth"]
    # 1 V1 + 3 features × 3 (V2, V3, V4) = 10 columns
    n_cols = 1 + 3 * 3
    n_rows = len(identity_results) + 1   # +1 for label row
    sheet = np.full((n_rows * tile_size, n_cols * tile_size, 3), 255,
                    dtype=np.uint8)
    # Label row
    labels = ["V1\norig"] + sum(
        [[f"V2_{f}\nwhole_foil", f"V3_{f}\npart_targ", f"V4_{f}\npart_foil"]
         for f in feature_order], [])
    for c, txt in enumerate(labels):
        canvas = np.full((tile_size, tile_size, 3), 230, dtype=np.uint8)
        for li, line in enumerate(txt.split("\n")):
            cv2.putText(canvas, line, (10, 40 + li * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        sheet[0:tile_size, c * tile_size:(c + 1) * tile_size] = canvas
    # Identity rows
    for r, (name, result) in enumerate(identity_results, start=1):
        # V1 column (any feature has identical V1)
        first_feat = result[feature_order[0]]
        v1 = cv2.resize(first_feat["V1"], (tile_size, tile_size))
        sheet[r * tile_size:(r + 1) * tile_size, 0:tile_size] = v1
        for f_i, feat in enumerate(feature_order):
            for v_i, k in enumerate(["V2", "V3", "V4"]):
                im = cv2.resize(result[feat][k], (tile_size, tile_size))
                c = 1 + f_i * 3 + v_i
                sheet[r * tile_size:(r + 1) * tile_size,
                      c * tile_size:(c + 1) * tile_size] = im
        # Identity label in left margin via overlay on V1
        cv2.putText(sheet, name,
                    (5, r * tile_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    Image.fromarray(sheet).save(out_path)


# === main (local 4-ID QC mode) ==============================================

def main(args):
    src_root = Path(args.source_dir)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    # Load identities
    id_dirs = sorted(src_root.glob("ffhq_*"))
    print(f"[init] found {len(id_dirs)} identity dirs in {src_root}")
    if len(id_dirs) < 4:
        print("[ERROR] need at least 4 identities for QC + donor pool")
        return
    embedder = FaceEmbedder(args.predictor, args.recognizer)
    all_attrs: list[IdAttr] = []
    for d in id_dirs:
        lm_path = d / "landmarks.json"; img_path = d / "thatcher_V1.png"
        if not (lm_path.exists() and img_path.exists()):
            continue
        meta = json.loads(lm_path.read_text())
        rgb = np.array(Image.open(img_path).convert("RGB"))
        lm = np.array(meta["landmarks"], dtype=np.float32)
        attr = build_attr(meta["ffhq_name"], rgb, lm, embedder)
        if attr is None:
            print(f"  [skip] {d.name}: no face detected by embedder")
            continue
        all_attrs.append(attr)
    print(f"[init] {len(all_attrs)} identities passed embedder")
    # Generate
    rng = random.Random(args.seed)
    n_qc = min(args.n_qc, len(all_attrs))
    qc_targets = all_attrs[:n_qc]
    results = []
    manifest = []
    for i in qc_targets:
        print(f"  [gen] {i.name} ...", end="", flush=True)
        out = partwhole_one_identity(i, all_attrs, rng)
        if out is None:
            print(" SKIP (no donors)"); continue
        results.append((i.name, out))
        # Save per-id PNGs
        idir = out_root / f"ffhq_{i.name}"; idir.mkdir(parents=True, exist_ok=True)
        for feat, r in out.items():
            for vk in ["V1", "V2", "V3", "V4"]:
                p = idir / f"partwhole_{feat}_{vk}.png"
                Image.fromarray(r[vk]).save(p)
                manifest.append({
                    "ffhq_name": i.name, "feature": feat, "condition": vk,
                    "donor_name": r["donor_name"],
                    "ssim_v1v2_bbox": r["meta"]["ssim_v1v2_bbox"],
                    "path": str(p),
                })
        print(f" donors: eye={out['eye']['donor_name']}, "
              f"nose={out['nose']['donor_name']}, mouth={out['mouth']['donor_name']}")
    # Manifest
    if manifest:
        with open(out_root / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader(); w.writerows(manifest)
    # QC contact sheet
    if results:
        qc_path = out_root / "qc_partwhole.png"
        make_qc_sheet(results, qc_path)
        print(f"\n[done] {len(results)} identities × 3 features × 4 conds")
        print(f"  QC sheet: {qc_path}")
        print(f"  manifest:  {out_root / 'manifest.csv'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", default="data/local_qc_partwhole",
                   help="dir containing ffhq_* subdirs with thatcher_V1.png + landmarks.json")
    p.add_argument("--output_dir", default="data/local_partwhole_v1")
    p.add_argument("--predictor", default="models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--recognizer", default="models/dlib_face_recognition_resnet_model_v1.dat")
    p.add_argument("--n_qc", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260526)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
