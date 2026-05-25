"""stimuli/classical_cv/template_partwhole.py — Tanaka 2004 template-based PW.

Faithful adaptation of Tanaka et al. (2004) / Crookes et al. (2013) PW
stimulus construction:
  "Within each of these sex and race categories a single face outline
   template was used. Each target face was created by pasting eyes, nose,
   and mouth features (taken from three different individuals) into the
   template. Foils for the 'whole' condition were created by swapping one
   feature in the target face with that of another target face of the
   same race and sex."

Pipeline (per template T from demographic cluster D):
  1. Pick K donor faces from D's cluster (~10-20)
  2. Build N_TARGETS target faces; each target i = T outline + features
     from 3 distinct donors {eye=A_i, nose=B_i, mouth=C_i}
     - For each feature: TPS-warp donor feature to T's landmark positions,
       Reinhard recolor to T's skin, Poisson seamlessClone into T
  3. For each target × each feature:
     V1 = whole target (T + this target's features)
     V2 = whole foil  (T + same features except this one swapped with
                       another target's same feature)
     V3 = target's feature isolated on gray (extracted from V1)
     V4 = foil's feature isolated on gray (extracted from V2) — SAME bbox

Per template: N_TARGETS × 3 features × 4 conditions = e.g., 6 × 3 × 4 = 72 PNG.

Why this works (vs naive cross-identity FFHQ-26k pairings):
  - All features paste into the SAME template outline → no inter-person
    face-shape mismatch at boundary
  - Features TPS-warped to template's landmark positions → no proportion
    mismatch ("nose too big/small for this face")
  - Reinhard normalizes skin tone before Poisson → no color seam
  - Demographic clustering ensures features are anatomically plausible
    on template (within-race within-sex)
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from stimuli.classical_cv.color_transfer import reinhard_transfer
from stimuli.classical_cv.precise_polygons import (
    both_eyes_polygon, nose_polygon, mouth_polygon,
    both_eyes_tight_polygon, nose_tight_polygon, mouth_tight_polygon,
    polygon_centroid, polygon_bbox, polygon_mask,
)
from stimuli.classical_cv.feature_warp import tps_warp_image
from stimuli.classical_cv.face_embedding import FaceEmbedder
from stimuli.classical_cv.donor_match import (
    build_attr, IdAttr, is_likely_infant,
)


FEATURES = ["eye", "nose", "mouth"]


def _build_padded_polygon(feature: str, lm: np.ndarray,
                           iod: float) -> np.ndarray:
    if feature == "eye":   return both_eyes_polygon(lm, iod=iod)
    if feature == "nose":  return nose_polygon(lm, iod=iod)
    if feature == "mouth": return mouth_polygon(lm, iod=iod)
    raise ValueError(feature)


def _build_tight_polygon(feature: str, lm: np.ndarray,
                          iod: float) -> np.ndarray:
    if feature == "eye":   return both_eyes_tight_polygon(lm)
    if feature == "nose":  return nose_tight_polygon(lm, iod=iod)
    if feature == "mouth": return mouth_tight_polygon(lm)
    raise ValueError(feature)


def _feature_landmark_indices(feature: str) -> list[int]:
    """The landmark subset that DEFINES this feature's position. Used as
    TPS control points (others are anchor points). Includes the union of
    landmarks bordering the feature so the warp doesn't kink at edges."""
    if feature == "eye":
        # left eye 36-41, right eye 42-47, plus brow lines as boundary anchors
        return list(range(36, 48))
    if feature == "nose":
        return list(range(27, 36))           # bridge + base
    if feature == "mouth":
        return list(range(48, 60))           # outer lip
    raise ValueError(feature)


def _annulus_skin_pixels(rgb: np.ndarray, mask: np.ndarray,
                          ring_px: int = 30) -> np.ndarray:
    dilated = cv2.dilate(mask, np.ones((ring_px, ring_px), np.uint8))
    ring = (dilated > 0) & (mask == 0)
    if not ring.any():
        return rgb.reshape(-1, 3)[:1000]
    return rgb[ring]


def _feature_anchor_pairs(feature: str, lm: np.ndarray
                          ) -> np.ndarray:
    """Return 2 anchor points per feature for similarity transform.
    Using ONLY 2 anchors means cv2.estimateAffinePartial2D computes a
    similarity (translation + rotation + isotropic scale) — preserving
    the feature's shape, only repositioning + rescaling."""
    if feature == "eye":
        # Left eye center + right eye center → spans the eye line
        return np.float32([lm[36:42].mean(0), lm[42:48].mean(0)])
    if feature == "nose":
        # Bridge top + bridge tip → spans the nose centerline
        return np.float32([lm[27], lm[30]])
    if feature == "mouth":
        # Left + right outer mouth corners
        return np.float32([lm[48], lm[54]])
    raise ValueError(feature)


def _warp_donor_to_template(donor_rgb: np.ndarray, donor_lm: np.ndarray,
                             template_lm: np.ndarray, feature: str
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Similarity transform (NOT TPS) aligning donor's feature anchors to
    template's. Preserves the donor's feature SHAPE while repositioning
    + rescaling it to fit template's feature socket.

    Why similarity instead of TPS:
      - TPS with all 68 landmarks distorts the donor's nose to match
        template's nose shape, erasing the identity-distinctive shape
        we're trying to swap in. Tanaka 2004 paste features unaltered.
      - Similarity (2 anchors) preserves shape; only translates+rotates+
        uniformly scales. Donor's nose stays donor-shaped, just sits in
        template's nose location.
    """
    src = _feature_anchor_pairs(feature, donor_lm)
    dst = _feature_anchor_pairs(feature, template_lm)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:
        M = np.eye(2, 3, dtype=np.float32)
    h, w = donor_rgb.shape[:2]
    warped = cv2.warpAffine(donor_rgb, M, (w, h),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE)
    # Warp donor landmarks too
    lm_h = np.concatenate([donor_lm, np.ones((68, 1))], axis=1)
    lm_warped = (M @ lm_h.T).T
    return warped, lm_warped.astype(np.float32)


def composite_template_face(template_rgb: np.ndarray, template_lm: np.ndarray,
                             donors: dict[str, tuple[np.ndarray, np.ndarray]],
                             iod_t: float) -> np.ndarray:
    """Build a target face by pasting `donors[feature]`'s feature into
    `template`. Returns the composite RGB.

    `donors` = {feature: (donor_rgb, donor_lm)} for each of eye/nose/mouth.
    The 3 donor faces are distinct (Tanaka design)."""
    h, w = template_rgb.shape[:2]
    out = template_rgb.copy()
    for feature in FEATURES:
        donor_rgb, donor_lm = donors[feature]
        # 1. Similarity-warp donor so its feature anchors align with template's
        warped_rgb, lm_warped = _warp_donor_to_template(
            donor_rgb, donor_lm, template_lm, feature)
        # 2. Build polygons — union of template's polygon AND warped-donor's
        #    polygon (ghost-union: captures both shapes, so the swap covers
        #    all of donor's distinctive shape even if larger/smaller than
        #    template's feature)
        poly_template = _build_padded_polygon(feature, template_lm, iod_t)
        poly_donor    = _build_padded_polygon(feature, lm_warped, iod_t)
        mask_template = polygon_mask(poly_template, (h, w))
        mask_donor    = polygon_mask(poly_donor, (h, w))
        mask_union    = cv2.bitwise_or(mask_template, mask_donor)
        # Center of union
        M = cv2.moments(mask_union)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = polygon_centroid(poly_template)
        # 3. Reinhard: recolor warped donor → match local skin annulus on `out`
        skin_ref = _annulus_skin_pixels(out, mask_union, ring_px=30)
        warped_recolor = reinhard_transfer(warped_rgb,
                                           skin_ref[None, :, :])
        # 4. Poisson seamlessClone — MIXED_CLONE preserves donor's strong
        #    gradients (eyelashes, iris edges, lip lines) better than
        #    NORMAL_CLONE, reducing the "soft halo" around the swap.
        try:
            poisson = cv2.seamlessClone(warped_recolor, out, mask_union,
                                        (cx, cy), cv2.MIXED_CLONE)
        except cv2.error:
            alpha = cv2.GaussianBlur(mask_union, (0, 0), 5, 5).astype(np.float32) / 255.0
            alpha = alpha[..., None]
            poisson = (out.astype(np.float32) * (1 - alpha) +
                       warped_recolor.astype(np.float32) * alpha).astype(np.uint8)
        # 5. Detail-preservation layer: Poisson attenuates high-freq details
        #    (eyelashes, iris texture, lip lines) near mask boundary. Add
        #    back donor's high-frequency content INSIDE the tight feature
        #    polygon — restores crisp detail without re-introducing the seam.
        poly_tight = _build_tight_polygon(feature, template_lm, iod_t)
        mask_tight = polygon_mask(poly_tight, (h, w))
        # Erode tight mask inward so detail layer doesn't reach the Poisson
        # transition zone (avoid double-edge artifacts)
        mask_inner = cv2.erode(mask_tight, np.ones((4, 4), np.uint8))
        alpha_inner = cv2.GaussianBlur(mask_inner, (0, 0), 2.0)
        alpha_inner = (alpha_inner.astype(np.float32) / 255.0)[..., None]
        donor_low = cv2.GaussianBlur(warped_recolor, (0, 0), 2.0)
        donor_detail = warped_recolor.astype(np.float32) - donor_low.astype(np.float32)
        out_f = poisson.astype(np.float32) + donor_detail * alpha_inner
        out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def isolate_feature_on_gray(rgb: np.ndarray, template_lm: np.ndarray,
                             feature: str, iod_t: float) -> np.ndarray:
    """Extract feature region (using TEMPLATE's tight polygon) from rgb,
    composite on gray=128. Tight polygon is in template's frame, so this
    gives a position-fixed crop for V3/V4."""
    h, w = rgb.shape[:2]
    poly_tight = _build_tight_polygon(feature, template_lm, iod_t)
    mask_tight = polygon_mask(poly_tight, (h, w))
    gray = np.full_like(rgb, 128)
    alpha = cv2.GaussianBlur(mask_tight, (0, 0), 2, 2).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    return (gray.astype(np.float32) * (1 - alpha) +
            rgb.astype(np.float32) * alpha).astype(np.uint8)


# === Build target faces + PW conditions for one template ====================

def build_template_stimuli(template: IdAttr, donor_pool: list[IdAttr],
                           rng: random.Random,
                           n_targets: int = 6) -> list[dict]:
    """Generate `n_targets` target faces under this template, plus the
    PW V1-V4 conditions for each (target × feature) combination.

    Returns list of dicts, one per (target_idx, feature)."""
    iod_t = float(np.linalg.norm(
        template.lm[36:42].mean(0) - template.lm[42:48].mean(0)))
    if len(donor_pool) < 6:
        return []
    # Build n_targets target faces. Each target = 3 distinct donors.
    targets = []  # list of {eye_donor, nose_donor, mouth_donor, composite_rgb}
    for t_idx in range(n_targets):
        # Pick 3 distinct donors (different from previous targets when possible)
        triple = rng.sample(donor_pool, 3)
        donors = {
            "eye":   (triple[0].rgb, triple[0].lm),
            "nose":  (triple[1].rgb, triple[1].lm),
            "mouth": (triple[2].rgb, triple[2].lm),
        }
        composite = composite_template_face(template.rgb, template.lm,
                                            donors, iod_t)
        targets.append({
            "donor_names": {f: triple[i].name for i, f in enumerate(FEATURES)},
            "donor_attrs": {f: triple[i] for i, f in enumerate(FEATURES)},
            "composite":   composite,
        })

    # Build PW conditions
    results = []
    for t_idx, t in enumerate(targets):
        # Pick foil = another target index (not t_idx)
        foil_options = [j for j in range(n_targets) if j != t_idx]
        foil_idx = rng.choice(foil_options)
        foil = targets[foil_idx]
        for feature in FEATURES:
            # V2 = T + t's features except `feature` swapped with foil's
            v2_donors = {
                "eye":   t["donor_attrs"]["eye"],
                "nose":  t["donor_attrs"]["nose"],
                "mouth": t["donor_attrs"]["mouth"],
            }
            v2_donors[feature] = foil["donor_attrs"][feature]
            v2_composite = composite_template_face(
                template.rgb, template.lm,
                {f: (a.rgb, a.lm) for f, a in v2_donors.items()},
                iod_t,
            )
            v1 = t["composite"]
            v2 = v2_composite
            v3 = isolate_feature_on_gray(v1, template.lm, feature, iod_t)
            v4 = isolate_feature_on_gray(v2, template.lm, feature, iod_t)
            # SSIM in feature bbox
            poly_t = _build_tight_polygon(feature, template.lm, iod_t)
            bx1, by1, bx2, by2 = polygon_bbox(poly_t, v1.shape[:2], pad=6)
            ssim_val = (float(ssim(v1[by1:by2, bx1:bx2],
                                   v2[by1:by2, bx1:bx2],
                                   channel_axis=2, data_range=255))
                        if (by2 > by1 and bx2 > bx1) else 1.0)
            results.append({
                "template": template.name,
                "target_idx": t_idx,
                "feature": feature,
                "target_donor_name": t["donor_names"][feature],
                "foil_donor_name":   foil["donor_names"][feature],
                "ssim_v1v2": ssim_val,
                "V1": v1, "V2": v2, "V3": v3, "V4": v4,
            })
    return results


# === Template selection via demographic clustering ==========================

def pick_templates_and_pools(all_attrs: list[IdAttr],
                              n_templates: int,
                              donors_per_template: int,
                              rng: random.Random
                              ) -> list[tuple[IdAttr, list[IdAttr]]]:
    """Cluster all_attrs into n_templates demographic groups via k-means on
    [skin_L, skin_a, skin_b, face_ratio, lwr_upr_ratio, nose_rel]. For
    each cluster: pick the most-central face as template + sample others
    as donor pool."""
    feats = np.array([
        [a.skin_lab[0], a.skin_lab[1], a.skin_lab[2],
         a.face_ratio, a.lwr_upr_ratio, a.nose_rel]
        for a in all_attrs
    ], dtype=np.float32)
    # Normalize
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    # cv2 kmeans
    K = min(n_templates, len(all_attrs) // 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        feats, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    out = []
    for k in range(K):
        members = np.where(labels == k)[0]
        if len(members) < 7:
            continue
        # Most central member = template
        dists = np.linalg.norm(feats[members] - centers[k], axis=1)
        template_idx = int(members[np.argmin(dists)])
        # Donor pool = up to donors_per_template OTHER members
        donor_indices = [int(i) for i in members if i != template_idx]
        rng.shuffle(donor_indices)
        donor_indices = donor_indices[:donors_per_template]
        out.append((all_attrs[template_idx],
                    [all_attrs[i] for i in donor_indices]))
    return out


# === QC sheet ===============================================================

def make_qc_sheet(per_template_results: list[tuple[str, list[dict]]],
                   out_path: Path, tile_size: int = 220) -> None:
    """Rows = (template, target_idx, feature); cols = V1, V2, V3, V4."""
    rows_data = []
    for tname, results in per_template_results:
        for r in results:
            rows_data.append((tname, r))
    n_cols = 4
    n_rows = len(rows_data) + 1
    sheet = np.full((n_rows * tile_size, n_cols * tile_size, 3), 255,
                    dtype=np.uint8)
    # Header
    for c, txt in enumerate(["V1 target", "V2 foil", "V3 part_t", "V4 part_f"]):
        canvas = np.full((tile_size, tile_size, 3), 230, dtype=np.uint8)
        cv2.putText(canvas, txt, (5, tile_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        sheet[0:tile_size, c * tile_size:(c + 1) * tile_size] = canvas
    for r_idx, (tname, r) in enumerate(rows_data, start=1):
        for c_idx, vk in enumerate(["V1", "V2", "V3", "V4"]):
            tile = cv2.resize(r[vk], (tile_size, tile_size))
            sheet[r_idx * tile_size:(r_idx + 1) * tile_size,
                  c_idx * tile_size:(c_idx + 1) * tile_size] = tile
        # Label on V1 column
        lbl = f"T={tname[:5]}/t{r['target_idx']}/{r['feature']}"
        cv2.putText(sheet, lbl, (5, r_idx * tile_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    Image.fromarray(sheet).save(out_path)


# === Main ===================================================================

def main(args):
    src_root = Path(args.source_dir)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    id_dirs = sorted(src_root.glob("ffhq_*"))
    print(f"[init] {len(id_dirs)} identity dirs in {src_root}")
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
        if attr is None or is_likely_infant(attr):
            continue
        all_attrs.append(attr)
    print(f"[init] {len(all_attrs)} adult identities in pool")
    rng = random.Random(args.seed)
    template_groups = pick_templates_and_pools(
        all_attrs, n_templates=args.n_templates,
        donors_per_template=args.donors_per_template, rng=rng)
    print(f"[init] {len(template_groups)} templates picked (k-means clustering)")
    per_template_results = []
    manifest = []
    for template, donor_pool in template_groups:
        print(f"  [gen] template={template.name} donors={len(donor_pool)}")
        results = build_template_stimuli(template, donor_pool, rng,
                                         n_targets=args.n_targets)
        if not results:
            print(f"    skip: too few donors")
            continue
        per_template_results.append((template.name, results))
        # Save PNGs
        tdir = out_root / f"template_{template.name}"
        tdir.mkdir(parents=True, exist_ok=True)
        for r in results:
            base = f"target{r['target_idx']:02d}_{r['feature']}"
            for vk in ["V1", "V2", "V3", "V4"]:
                p = tdir / f"{base}_{vk}.png"
                Image.fromarray(r[vk]).save(p)
                manifest.append({
                    "template": template.name,
                    "target_idx": r["target_idx"],
                    "feature": r["feature"],
                    "condition": vk,
                    "target_donor": r["target_donor_name"],
                    "foil_donor": r["foil_donor_name"],
                    "ssim_v1v2": r["ssim_v1v2"],
                    "path": str(p),
                })
    # Manifest
    if manifest:
        with open(out_root / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader(); w.writerows(manifest)
    # QC sheet (cap rows to 30 to keep image readable)
    short_results = []
    for tname, results in per_template_results:
        short_results.append((tname, results[:6]))  # first 6 rows per template
    if short_results:
        qc_path = out_root / "qc_template_partwhole.png"
        make_qc_sheet(short_results, qc_path)
        print(f"\n[done] {len(per_template_results)} templates × "
              f"{args.n_targets} targets × 3 features × 4 conds")
        print(f"  QC sheet: {qc_path}")
        print(f"  manifest: {out_root / 'manifest.csv'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", default="data/local_qc_partwhole")
    p.add_argument("--output_dir", default="data/local_template_partwhole_v1")
    p.add_argument("--predictor",
                   default="models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--recognizer",
                   default="models/dlib_face_recognition_resnet_model_v1.dat")
    p.add_argument("--n_templates", type=int, default=5)
    p.add_argument("--donors_per_template", type=int, default=10)
    p.add_argument("--n_targets", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260526)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
