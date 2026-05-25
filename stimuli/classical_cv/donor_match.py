"""stimuli/classical_cv/donor_match.py — per-feature donor matching.

For each recipient identity i, picks an independent donor for each feature:
  j_eye, j_nose, j_mouth — possibly different identities.

Strategy:
  HARD CONSTRAINTS (filter the pool):
    - skin tone LAB-L distance ≤ lab_L_max  (avoid different ethnicity)
    - face proportion ratio diff ≤ face_ratio_max  (similar face shape)
    - dlib face-embedding L2 ∈ [d_low, d_high]
        d_low > 0.55 → not same person (avoid V1 ≈ V2)
        d_high < 0.85 → still recognizably similar (avoid jarring swap)

  RANDOMIZED PICK from filtered pool — relying on hard constraints +
  the post-hoc SSIM reject (in partwhole_classical) to handle outliers.
  This is simpler than ranking by feature-distance and avoids the
  failure mode where the "most similar nose" donor produces V1 ≈ V2.

Attribute extraction:
  - skin LAB: cheek annulus (face-oval mask minus eye/brow/mouth)
  - face ratio: face_width / face_height from jaw + brow landmarks
  - nose relative width: (lm 35.x - lm 31.x) / face_width
  - dlib face embedding: 128-d ResNet descriptor
"""
from __future__ import annotations
import random
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class IdAttr:
    name: str
    rgb: np.ndarray
    lm: np.ndarray
    emb: np.ndarray          # 128-d
    skin_lab: np.ndarray     # (3,) mean LAB
    face_w: float
    face_h: float
    face_ratio: float        # face_w / face_h
    nose_rel: float          # nose_width / face_w
    iod_face_ratio: float    # IOD / face_w (not strongly discriminative across ages)
    cheek_smoothness: float  # Laplacian var; unreliable as absolute age proxy
    lwr_upr_ratio: float     # (chin_y - eye_y) / (eye_y - brow_y) — INFANT INDICATOR
                              # infants/toddlers < 4.0; adults typically 4.5-7.5


def is_likely_infant(attr: "IdAttr") -> bool:
    """Identify babies/toddlers via (chin-to-eye)/(eye-to-brow) ratio.
    Infants/toddlers: ratio < 3.85 (eyes sit relatively low in face due to
    short chin + large forehead). Adults: typically 4+.

    Calibrated on 41-ID FFHQ pool 2026-05-26: catches babies/toddlers
    (00003 baby 3.60, 00010 toddler 3.71, 04500 toddler 2.97, 55000 infant
    3.29). Cost: ~1 adult false positive (00002 adult Asian woman 3.78,
    unusually short-chinned). Acceptable at 26k scale — we'd rather lose
    a few borderline adults than include any infant whose anatomy breaks
    cross-identity matching.
    """
    return attr.lwr_upr_ratio < 3.85


def _skin_mean_lab(rgb: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Mean LAB of skin: jaw polygon - (eyes ∪ brows ∪ mouth ∪ nostrils)."""
    h, w = rgb.shape[:2]
    jaw = lm[0:17].astype(np.int32)
    # Build face mask: convex hull of jaw + estimated forehead
    brow_top_y = float(lm[17:27, 1].min())
    iod = float(np.linalg.norm(lm[36:42].mean(0) - lm[42:48].mean(0)))
    forehead_y = max(0.0, brow_top_y - 0.20 * iod)
    fl = np.array([jaw[0, 0], forehead_y], dtype=np.int32)
    fr = np.array([jaw[-1, 0], forehead_y], dtype=np.int32)
    face_poly = np.concatenate([jaw, fr[None, :], fl[None, :]], axis=0)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, cv2.convexHull(face_poly.astype(np.int32)), 255)
    # Subtract eye + brow + mouth + nostril regions
    sub = np.zeros_like(face_mask)
    for pts in [lm[17:27], lm[36:48], lm[48:68], lm[31:36]]:
        cv2.fillConvexPoly(sub, cv2.convexHull(pts.astype(np.int32)), 255)
    skin_mask = (face_mask > 0) & (sub == 0)
    if not skin_mask.any():
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[skin_mask].mean(axis=0).astype(np.float32)


def _cheek_smoothness(rgb: np.ndarray, lm: np.ndarray, iod: float) -> float:
    """Laplacian variance in cheek skin region. Higher = more high-frequency
    energy (wrinkles, texture) = older. Lower = smoother = younger.

    Cheek region heuristic: below the eye-line, above the mouth-line,
    inside jaw landmarks 2..14.
    """
    h, w = rgb.shape[:2]
    cheek_y1 = int(lm[36:48, 1].mean() + 0.30 * iod)
    cheek_y2 = int(lm[48:68, 1].min() - 0.10 * iod)
    cheek_x1 = int(max(0, lm[2, 0]))
    cheek_x2 = int(min(w, lm[14, 0]))
    if cheek_y2 <= cheek_y1 + 5 or cheek_x2 <= cheek_x1 + 5:
        return 0.0
    crop = cv2.cvtColor(rgb[cheek_y1:cheek_y2, cheek_x1:cheek_x2],
                        cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    return float(lap.var())


def build_attr(name: str, rgb: np.ndarray, lm: np.ndarray,
               embedder) -> IdAttr | None:
    """Compute attribute vector for one identity. Returns None if no face detected."""
    emb, _shape = embedder.embed(rgb)
    if emb is None:
        return None
    skin_lab = _skin_mean_lab(rgb, lm)
    face_w = float(lm[16, 0] - lm[0, 0])
    brow_y = float((lm[19, 1] + lm[24, 1]) / 2.0)
    face_h = float(lm[8, 1] - brow_y)
    face_ratio = face_w / max(face_h, 1e-6)
    nose_w = float(lm[35, 0] - lm[31, 0])
    nose_rel = nose_w / max(face_w, 1e-6)
    iod = float(np.linalg.norm(lm[36:42].mean(0) - lm[42:48].mean(0)))
    iod_face_ratio = iod / max(face_w, 1e-6)
    smoothness = _cheek_smoothness(rgb, lm, iod)
    # Infant-discriminating proportion: ratio of (chin-to-eye)/(eye-to-brow).
    # Infants have shorter chins + larger foreheads → ratio is LOW.
    eye_y = float(lm[36:48, 1].mean())
    chin_y = float(lm[8, 1])
    lwr = chin_y - eye_y
    upr = eye_y - brow_y
    lwr_upr_ratio = lwr / max(upr, 1e-6)
    return IdAttr(
        name=name, rgb=rgb, lm=lm, emb=emb,
        skin_lab=skin_lab, face_w=face_w, face_h=face_h,
        face_ratio=face_ratio, nose_rel=nose_rel,
        iod_face_ratio=iod_face_ratio,
        cheek_smoothness=smoothness,
        lwr_upr_ratio=lwr_upr_ratio,
    )


def candidate_pool(i: IdAttr, all_attrs: list[IdAttr],
                   d_low: float = 0.55, d_high: float = 0.85,
                   lab_L_max: float = 8.0,
                   face_ratio_max: float = 0.10,
                   iod_ratio_max: float = 0.05,
                   smoothness_log_ratio_max: float = 0.30) -> list[IdAttr]:
    """Filter all_attrs to those passing the hard constraints w.r.t. i.

    Age-related constraints (NEW):
      iod_ratio_max: |i.iod/face_w - j.iod/face_w| ≤ 0.05
        Children's IOD/face_w ≈ 0.45-0.55; adults ≈ 0.30-0.40.
      smoothness_log_ratio_max: |log10(i.smoothness+1) - log10(j.smoothness+1)|
        ≤ 0.7  (i.e. <5x ratio either way). Log-space because cheek
        variance spans 1-3 orders of magnitude across faces.
    """
    out = []
    for j in all_attrs:
        if j.name == i.name:
            continue
        if abs(i.skin_lab[0] - j.skin_lab[0]) > lab_L_max:
            continue
        if abs(i.face_ratio - j.face_ratio) > face_ratio_max:
            continue
        if abs(i.iod_face_ratio - j.iod_face_ratio) > iod_ratio_max:
            continue
        ls_i = float(np.log10(i.cheek_smoothness + 1.0))
        ls_j = float(np.log10(j.cheek_smoothness + 1.0))
        if abs(ls_i - ls_j) > smoothness_log_ratio_max:
            continue
        d_emb = float(np.linalg.norm(i.emb - j.emb))
        if not (d_low <= d_emb <= d_high):
            continue
        out.append(j)
    return out


def pick_donors(i: IdAttr, all_attrs: list[IdAttr],
                rng: random.Random,
                d_low: float = 0.55, d_high: float = 0.85,
                lab_L_max: float = 8.0,
                face_ratio_max: float = 0.15) -> dict[str, IdAttr | None]:
    """Pick three INDEPENDENT donors (eye/nose/mouth). Returns dict with
    keys 'eye', 'nose', 'mouth' (each value may be None if pool empty).

    If the pool has ≥ 3 candidates, we sample 3 DISTINCT donors so that
    each feature swap exposes a different "other identity" — this is the
    natural Tanaka-style design (the foil for one feature need not be the
    foil for another).
    """
    pool = candidate_pool(i, all_attrs, d_low, d_high, lab_L_max, face_ratio_max)
    if not pool:
        return {"eye": None, "nose": None, "mouth": None}
    if len(pool) >= 3:
        picks = rng.sample(pool, 3)
    else:
        picks = [rng.choice(pool) for _ in range(3)]
    return {"eye": picks[0], "nose": picks[1], "mouth": picks[2]}


def relax_pool_if_empty(i: IdAttr, all_attrs: list[IdAttr], rng: random.Random,
                        d_low_start: float = 0.55, d_high_start: float = 0.85,
                        lab_L_max_start: float = 8.0,
                        face_ratio_max_start: float = 0.15,
                        relax_steps: int = 3) -> dict[str, IdAttr | None]:
    """Try pick_donors with progressive constraint relaxation.

    Each step expands the embedding window by ±0.05 and skin tone by +3.0
    LAB units. Stops at first non-empty pool, or returns Nones."""
    for step in range(relax_steps):
        d_low = max(0.0, d_low_start - 0.05 * step)
        d_high = d_high_start + 0.05 * step
        lab_L = lab_L_max_start + 3.0 * step
        face_ratio = face_ratio_max_start + 0.05 * step
        donors = pick_donors(i, all_attrs, rng, d_low, d_high, lab_L, face_ratio)
        if donors["eye"] is not None:
            return donors
    return {"eye": None, "nose": None, "mouth": None}
