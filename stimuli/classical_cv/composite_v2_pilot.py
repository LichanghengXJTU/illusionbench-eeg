"""stimuli/classical_cv/composite_v2_pilot.py — Young 1987 composite (v2.0) pilot.

Self-contained pilot for the Composite Face Illusion subset of
HoloFaceIllusion-Bench-EEG. Runs on local Mac without server-side
attrs.pkl — loads the 111 pre-filtered FFHQ identities under
illusionbench/data/local_qc_partwhole/, recomputes mini-attrs, and
generates 4 conditions per (template, donor) pair using:

    V1  aligned upright  (T_top + D_bottom_warped, aligned)
    V2  misaligned upright (V1 with bottom shifted dx = 0.5 * face_w)
    V3  aligned inverted  (V1 rotated 180°)
    V4  misaligned inverted (V2 rotated 180°)

Canonical paradigm (Young, Hellawell & Hay 1987; McKone et al. 2013;
Murphy, Gray & Cook 2017):
  - Top-half of template T + bottom-half of donor D (different identities)
  - Horizontal cut at upper-lip / nose-bottom midpoint:
      y_cut = (lm[33].y + lm[51].y) / 2
  - Aligned-upright induces illusion of "novel facial identity";
    misalignment and inversion are controls that null the illusion.

Algorithm:
  Phase A (per identity, ~111 local faces):
    dlib 68 landmarks (cached in landmarks.json) + dlib 128-d face embedding
    + skin LAB mean + face_w / bottom_face_w geometry.

  Phase B (per template):
    Filter donor pool:
      skin LAB Δ ≤ (10, 8, 8)  (looser than full-scale for pilot's 111-pool)
      embed L2 ∈ [0.55, 0.85]   (different identity, same demographic)
      |bottom_w / face_w Δ| ≤ 0.07
    Pick K=3 donors via farthest-point sampling on dlib embedding
      (distinct perceptual identities, not 3 near-duplicates).
    For each donor:
      - TPS-warp donor bottom (lm 3-14 jaw + 31-35 nose base + 48-67 mouth)
        from donor coords to template coords; top-half landmarks act as anchors.
      - Cut at y_cut and stitch T_top + D_warped_bottom.
      - Reinhard LAB ring color transfer (L full, a/b 50% blend) using a
        30-px ring above + below cut line as anchor.
      - Poisson NORMAL_CLONE seam blend in a ±8-px band around y_cut.
      - Build face oval mask from T.lm (jaw 0-16 + interpolated forehead arc).
        Pixels outside the oval → mid-gray (128, 128, 128).
      - Generate V1/V2/V3/V4.
      - Write 4 PNGs + 1 JSON per (T, D).

  Phase C (QC):
    Contact sheet (4 templates × 4 conditions = 16 panels per page)
    + summary metrics: SSIM(V1, V2) in face oval, seam_disc at cut ± 3 px,
    LAB delta at cut ring.

For full-scale generation (~20k templates) we'll:
  - Replace mini Phase A with the parallel attrs pipeline that uses
    InsightFace gender hard-match + CelebA facial-hair binary match.
  - Use multiprocessing pool keyed on template index.
  - Pack into webdataset tars under v2.0/composite_NNN.tar.

Usage:
    cd ~/Desktop/EEG/illusionbench
    source .venv/bin/activate
    python -m stimuli.classical_cv.composite_v2_pilot \\
        --source_dir data/local_qc_partwhole \\
        --output_dir data/composite_pilot \\
        --n_templates 8 \\
        --donors_per_template 3
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from stimuli.classical_cv.face_embedding import FaceEmbedder
from stimuli.classical_cv.color_transfer import reinhard_L_only_ring
from stimuli.classical_cv.feature_warp import tps_warp_image
from stimuli.classical_cv.insight_attrs import InsightAttrExtractor


# === Landmark groups (dlib 68-point convention) ===
JAW           = list(range(0, 17))
LEFT_BROW     = list(range(17, 22))
RIGHT_BROW    = list(range(22, 27))
NOSE_BRIDGE   = list(range(27, 31))
NOSE_BASE     = list(range(31, 36))
LEFT_EYE      = list(range(36, 42))
RIGHT_EYE     = list(range(42, 48))
MOUTH_OUTER   = list(range(48, 60))
MOUTH_INNER   = list(range(60, 68))

# Composite-specific landmark subsets used for similarity-transform alignment
EYE_ANCHOR_IDX = [LEFT_EYE, RIGHT_EYE]   # two eye-center anchors (similarity transform)

# === MediaPipe Face Mesh: 478 landmarks per face ====================
# FACE_OVAL is a well-known 36-landmark subset that traces the face boundary.
# Indices below are extracted from the canonical FACEMESH_FACE_OVAL
# connection set (Google MediaPipe spec, public). Ordered clockwise
# starting from top center, used to build a tight face mask.
MP_FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]   # 36 points
# The bottom HALF of the oval (from one ear-side down through chin to other
# ear-side) used to align cheek lines below the cut. Subject's right side
# (viewer's left): 234, 127 ... up to 10; subject's left side: 10 ... 365,
# 379, ..., 152, ..., back to 234. We pick points with normalized y > 0.5.
MP_BOTTOM_OVAL_IDX = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400,
    378, 379, 365, 397, 288, 361, 323, 454,
]   # ~21 points along the lower face contour, ordered left→right

# Background gray for face oval mask (matches PW v1.4 V3/V4 bg)
BG_GRAY = 128

# Cut line endpoints reference: nose-bottom (33) + upper-lip-center (51)
NOSE_BOTTOM_IDX = 33
UPPER_LIP_CENTER_IDX = 51


# === Data classes =====================================================

@dataclass
class IdAttr:
    name: str                       # ffhq_NNNNN
    rgb: np.ndarray                 # (H, W, 3) uint8
    lm: np.ndarray                  # (68, 2) float32 px coords (dlib)
    mp_lm: np.ndarray               # (478, 2) float32 px coords (MediaPipe Face Mesh)
    emb: np.ndarray                 # 128-d dlib embedding
    skin_lab: np.ndarray            # (3,) mean LAB
    face_w: float                   # px (lm[16].x - lm[0].x)
    face_h: float                   # px (lm[8].y - brow_top)
    bottom_w: float                 # px width at cut-line y
    bottom_face_w_ratio: float      # bottom_w / face_w  (target: ~0.5-0.7)
    y_cut: float                    # px y-coord of cut line
    lwr_upr_ratio: float            # (chin-y - eye-y) / (eye-y - brow-y) — infant proxy
    yaw_deg: float                  # head yaw (left/right rotation); 0 = frontal
    pitch_deg: float                # head pitch (up/down)
    roll_deg: float                 # head roll (tilt within image plane)
    mouth_offset_ratio: float       # |mouth_center_x - face_midline_x| / face_w
                                    # filter at 0.03 — bigger = mouth shifted off-axis
    cheek_w_at_cut: float           # px width of face oval at y_cut (MediaPipe precision)
    cut_to_chin_ratio: float        # (chin_y - y_cut) / face_w — bottom-face vertical extent
    predicted_age: int              # InsightFace age estimate; -1 if unavailable
    predicted_gender: str           # "M" / "F" / "?"


# === MediaPipe wrapper ===============================================

_MP_DETECTOR = None


def _get_mp_detector(model_path: str = "models/mediapipe/face_landmarker.task"):
    global _MP_DETECTOR
    if _MP_DETECTOR is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _MP_DETECTOR = mp_vision.FaceLandmarker.create_from_options(opts)
    return _MP_DETECTOR


def detect_mp_landmarks(rgb: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray] | None:
    """Run MediaPipe Face Mesh.

    Returns ((478, 2) pixel-coord landmarks, (4, 4) facial_transformation_matrix),
    or None if no face detected.
    """
    detector = _get_mp_detector()
    h, w = rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_img)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    arr = np.array([[l.x * w, l.y * h] for l in lms], dtype=np.float32)
    xform = np.array(result.facial_transformation_matrixes[0], dtype=np.float32) \
        if result.facial_transformation_matrixes else np.eye(4, dtype=np.float32)
    return arr, xform


def decompose_head_pose(xform: np.ndarray) -> tuple[float, float, float]:
    """Decompose 4x4 facial transformation matrix into yaw / pitch / roll degrees.

    MediaPipe's xform = T @ R applied to the canonical face mesh to produce
    the detected face in world space. R (top-left 3x3) is the head's rotation.

    Convention (right-handed):
      pitch = rotation around X (nodding up/down)
      yaw   = rotation around Y (turning left/right)
      roll  = rotation around Z (tilting within image plane)
    """
    R = xform[:3, :3]
    # Using the ZYX intrinsic (yaw-pitch-roll) decomposition
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        roll = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return float(yaw), float(pitch), float(roll)


# === Frontalization via auto-detected MediaPipe mirror pairs ============

def _auto_mirror_pairs(mp_lm: np.ndarray,
                        midline_x: float | None = None,
                        max_y_diff: float = 6.0
                        ) -> dict[int, int]:
    """For each landmark i, find the landmark j whose position best matches
    i's horizontal mirror across the face's vertical midline (with small y
    tolerance). Returns dict mapping i → j (and j → i). Self-pairs (center-
    line landmarks) are returned as i → i.

    midline_x: x-coordinate of face's vertical center line. Defaults to the
    midpoint between left and right inner eye corners (MediaPipe idx 133, 362).
    max_y_diff: pairs must have |y_i - y_j| ≤ this many pixels (rejects
    accidental cross-feature matches).
    """
    if midline_x is None:
        midline_x = float((mp_lm[133, 0] + mp_lm[362, 0]) / 2.0)
    n = len(mp_lm)
    pairs: dict[int, int] = {}
    for i in range(n):
        if i in pairs:
            continue
        target_x = 2 * midline_x - mp_lm[i, 0]
        target_y = mp_lm[i, 1]
        # Self-pair if landmark is on midline (within tolerance)
        if abs(mp_lm[i, 0] - midline_x) < 1.5:
            pairs[i] = i
            continue
        # Search for closest unmatched landmark
        best_j = -1
        best_d = float("inf")
        for j in range(n):
            if j == i or j in pairs:
                continue
            if abs(mp_lm[j, 1] - target_y) > max_y_diff:
                continue
            dx = mp_lm[j, 0] - target_x
            dy = mp_lm[j, 1] - target_y
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d; best_j = j
        if best_j >= 0:
            pairs[i] = best_j
            pairs[best_j] = i
        else:
            pairs[i] = i  # unmatched → keep in place
    return pairs


def frontalize_landmarks(mp_lm: np.ndarray) -> np.ndarray:
    """Compute a "frontal" version of mp_lm by averaging each landmark with
    its mirror counterpart. Yaw-induced left/right asymmetry cancels out;
    natural facial asymmetry is reduced (acceptable trade-off — we want
    a symmetric face for clean compositing).
    """
    midline_x = float((mp_lm[133, 0] + mp_lm[362, 0]) / 2.0)
    pairs = _auto_mirror_pairs(mp_lm, midline_x=midline_x)
    out = mp_lm.copy()
    visited: set[int] = set()
    for i, j in pairs.items():
        if i in visited or j in visited:
            continue
        if i == j:
            out[i, 0] = midline_x
            visited.add(i)
            continue
        # i is on one side; j is its mirror counterpart.
        # Frontal positions: average distance from midline; same y.
        d_i = mp_lm[i, 0] - midline_x      # signed offset of i from midline
        d_j = midline_x - mp_lm[j, 0]      # signed offset of j (mirrored sign)
        d_avg = (d_i + d_j) / 2.0
        y_avg = (mp_lm[i, 1] + mp_lm[j, 1]) / 2.0
        out[i] = (midline_x + d_avg, y_avg)
        out[j] = (midline_x - d_avg, y_avg)
        visited.add(i); visited.add(j)
    return out.astype(np.float32)


def frontalize_donor_image(donor_rgb: np.ndarray,
                            donor_mp_lm: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """TPS-warp donor's image so its landmarks become left-right symmetric.

    Returns (warped_rgb, frontalized_mp_lm). Use this BEFORE cheek-align +
    composite. Small yaws (≤ ~12°) are corrected; severe yaws warp the
    image too aggressively and create artifacts — filter those out instead.
    """
    front_lm = frontalize_landmarks(donor_mp_lm)
    warped = tps_warp_image(donor_rgb, donor_mp_lm, front_lm)
    return warped, front_lm


@dataclass
class CompositeResult:
    template_name: str
    donor_name: str
    y_cut: float
    face_w: float
    mis_dx_px: float                # misalignment shift used
    ssim_v1v2: float                # SSIM(V1, V2) in face oval
    seam_disc: float                # mean |∂I/∂y| in band cut±3 (lower=cleaner)
    lab_delta_cut: float            # |LAB ring T - LAB ring D| after color transfer
    v1: np.ndarray
    v2: np.ndarray
    v3: np.ndarray
    v4: np.ndarray


# === Phase A: per-identity attrs ======================================

def _skin_mean_lab(rgb: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Mean LAB color of face skin (face oval - eyes - brows - mouth - nostrils)."""
    h, w = rgb.shape[:2]
    jaw = lm[JAW].astype(np.int32)
    brow_top_y = float(lm[LEFT_BROW + RIGHT_BROW, 1].min())
    iod = float(np.linalg.norm(lm[LEFT_EYE].mean(0) - lm[RIGHT_EYE].mean(0)))
    forehead_y = max(0.0, brow_top_y - 0.20 * iod)
    fl = np.array([jaw[0, 0], forehead_y], dtype=np.int32)
    fr = np.array([jaw[-1, 0], forehead_y], dtype=np.int32)
    face_poly = np.concatenate([jaw, fr[None, :], fl[None, :]], axis=0)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, cv2.convexHull(face_poly.astype(np.int32)), 255)
    sub = np.zeros_like(face_mask)
    for pts in [lm[LEFT_BROW + RIGHT_BROW],
                lm[LEFT_EYE + RIGHT_EYE],
                lm[MOUTH_OUTER + MOUTH_INNER],
                lm[NOSE_BASE]]:
        cv2.fillConvexPoly(sub, cv2.convexHull(pts.astype(np.int32)), 255)
    skin = (face_mask > 0) & (sub == 0)
    if not skin.any():
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[skin].mean(axis=0).astype(np.float32)


def compute_attrs(name: str, rgb: np.ndarray, lm: np.ndarray,
                  embedder: FaceEmbedder,
                  insight: InsightAttrExtractor | None = None) -> IdAttr | None:
    emb, _ = embedder.embed(rgb)
    if emb is None:
        return None
    mp_out = detect_mp_landmarks(rgb)
    if mp_out is None:
        return None
    mp_lm, xform = mp_out
    yaw_deg, pitch_deg, roll_deg = decompose_head_pose(xform)
    skin_lab = _skin_mean_lab(rgb, lm)
    face_w = float(lm[16, 0] - lm[0, 0])
    brow_y = float((lm[19, 1] + lm[24, 1]) / 2.0)
    face_h = float(lm[8, 1] - brow_y)
    y_cut = float((lm[NOSE_BOTTOM_IDX, 1] + lm[UPPER_LIP_CENTER_IDX, 1]) / 2.0)
    # face width at the cut line (linearly interpolate left/right jaw)
    # Use jaw points 3-13 as a curve; sample x at y = y_cut.
    jaw = lm[JAW]
    # left side jaw 0-7, right side jaw 9-16 — pick points straddling y_cut
    left_x = _interp_jaw_x(jaw[:9], y_cut)
    right_x = _interp_jaw_x(jaw[8:], y_cut)
    bottom_w = float(right_x - left_x)
    bottom_face_w_ratio = bottom_w / max(face_w, 1e-6)
    eye_y = float(lm[LEFT_EYE + RIGHT_EYE, 1].mean())
    chin_y = float(lm[8, 1])
    lwr = chin_y - eye_y
    upr = eye_y - brow_y
    lwr_upr_ratio = lwr / max(upr, 1e-6)
    # Mouth-centrality: |mouth_center_x - face_midline_x| / face_w
    # MediaPipe: 13 = upper-lip-center, 14 = lower-lip-center; 133/362 = inner eye corners
    face_midline_x = float((mp_lm[133, 0] + mp_lm[362, 0]) / 2.0)
    mouth_center_x = float((mp_lm[13, 0] + mp_lm[14, 0]) / 2.0)
    mouth_offset_ratio = abs(mouth_center_x - face_midline_x) / max(face_w, 1e-6)
    # Precise face width at the cut line via MediaPipe face-oval contour
    cheek_x = cheek_x_at_y(mp_lm, y_cut)
    cheek_w_at_cut = (cheek_x[1] - cheek_x[0]) if cheek_x is not None else 0.0
    # Vertical distance from cut line to chin (normalized by face_w for size invariance)
    cut_to_chin_px = float(chin_y - y_cut)
    cut_to_chin_ratio = cut_to_chin_px / max(face_w, 1e-6)
    # InsightFace age / gender (optional)
    p_age = -1; p_gender = "?"
    if insight is not None:
        try:
            r = insight.predict(rgb, lm)
            if r is not None:
                p_age, p_gender = r
        except Exception:
            pass
    return IdAttr(name=name, rgb=rgb, lm=lm, mp_lm=mp_lm, emb=emb,
                  skin_lab=skin_lab, face_w=face_w, face_h=face_h,
                  bottom_w=bottom_w,
                  bottom_face_w_ratio=bottom_face_w_ratio,
                  y_cut=y_cut,
                  lwr_upr_ratio=lwr_upr_ratio,
                  yaw_deg=yaw_deg,
                  pitch_deg=pitch_deg,
                  roll_deg=roll_deg,
                  mouth_offset_ratio=mouth_offset_ratio,
                  cheek_w_at_cut=cheek_w_at_cut,
                  cut_to_chin_ratio=cut_to_chin_ratio,
                  predicted_age=p_age,
                  predicted_gender=p_gender)


def _interp_jaw_x(jaw_half: np.ndarray, y: float) -> float:
    """Given a jaw half (sequence of (x,y) points roughly monotonic in y),
    linearly interpolate x at the given y. Falls back to nearest point."""
    ys = jaw_half[:, 1]
    if y <= ys.min():
        return float(jaw_half[np.argmin(ys), 0])
    if y >= ys.max():
        return float(jaw_half[np.argmax(ys), 0])
    # find bracket
    for i in range(len(jaw_half) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            t = (y - y0) / (y1 - y0) if y1 != y0 else 0.0
            return float(jaw_half[i, 0] + t * (jaw_half[i + 1, 0] - jaw_half[i, 0]))
    return float(jaw_half[np.argmin(np.abs(ys - y)), 0])


def load_identities(src_dir: Path,
                    embedder: FaceEmbedder,
                    limit: int = 0,
                    infant_ratio_cutoff: float = 3.85,
                    yaw_abs_max: float = 15.0,
                    mouth_offset_max: float = 0.03,
                    insight: InsightAttrExtractor | None = None,
                    ) -> list[IdAttr]:
    """Load all ffhq_*/landmarks.json + thatcher_V1.png from src_dir and
    compute attrs. thatcher_V1.png is the original FFHQ face (the V1
    upright_normal Thatcher condition = unmodified).

    Infants/toddlers are filtered out (lwr_upr_ratio < cutoff): per
    [[project_partwhole_template_2026-05-26]] infant face proportions
    break cross-identity matching even with template anchoring."""
    out: list[IdAttr] = []
    n_infant = 0
    n_yaw = 0
    n_mouth = 0
    id_dirs = sorted(src_dir.glob("ffhq_*"))
    if limit:
        id_dirs = id_dirs[:limit]
    for d in id_dirs:
        lm_path = d / "landmarks.json"
        img_path = d / "thatcher_V1.png"
        if not (lm_path.exists() and img_path.exists()):
            continue
        try:
            meta = json.loads(lm_path.read_text())
            rgb = np.array(Image.open(img_path).convert("RGB"))
            lm = np.array(meta["landmarks"], dtype=np.float32)
            attr = compute_attrs(meta["ffhq_name"], rgb, lm, embedder, insight=insight)
            if attr is None:
                continue
            if attr.lwr_upr_ratio < infant_ratio_cutoff:
                n_infant += 1
                continue
            if abs(attr.yaw_deg) > yaw_abs_max:
                n_yaw += 1
                continue
            if attr.mouth_offset_ratio > mouth_offset_max:
                n_mouth += 1
                continue
            out.append(attr)
        except Exception as e:
            print(f"  skip {d.name}: {e}")
    print(f"  filtered {n_infant} infants/toddlers (lwr_upr_ratio < {infant_ratio_cutoff})")
    print(f"  filtered {n_yaw} non-frontal (|yaw| > {yaw_abs_max}°)")
    print(f"  filtered {n_mouth} mouth-off-axis (offset / face_w > {mouth_offset_max})")
    yaws = np.array([a.yaw_deg for a in out])
    print(f"  yaw distribution (kept N={len(out)}): "
          f"mean={yaws.mean():.1f}°  std={yaws.std():.1f}°  "
          f"min={yaws.min():.1f}°  max={yaws.max():.1f}°")
    return out


# === Phase B: donor matching ==========================================

def pick_donors(template: IdAttr, all_attrs: list[IdAttr],
                k: int = 3,
                lab_L_max: float = 10.0,
                lab_a_max: float = 8.0,
                lab_b_max: float = 8.0,
                d_emb_low: float = 0.55,
                d_emb_high: float = 0.85,
                cheek_w_pct_max: float = 0.10,
                yaw_diff_max: float = 10.0,
                age_diff_max: int = 15,
                require_gender_match: bool = True,
                cut_to_chin_pct_max: float = 0.10,
                ) -> list[IdAttr]:
    """Filter all_attrs to those matching template demographic, then pick K
    donors via farthest-point sampling on dlib embedding so the 3 chosen
    are perceptually distinct from each other."""
    pool = []
    for d in all_attrs:
        if d.name == template.name:
            continue
        if abs(d.skin_lab[0] - template.skin_lab[0]) > lab_L_max:
            continue
        if abs(d.skin_lab[1] - template.skin_lab[1]) > lab_a_max:
            continue
        if abs(d.skin_lab[2] - template.skin_lab[2]) > lab_b_max:
            continue
        d_emb = float(np.linalg.norm(d.emb - template.emb))
        if not (d_emb_low <= d_emb <= d_emb_high):
            continue
        # Cheek-width at cut line: percent-diff filter (user-requested 10%)
        if d.cheek_w_at_cut <= 0 or template.cheek_w_at_cut <= 0:
            continue
        cheek_pct = abs(d.cheek_w_at_cut - template.cheek_w_at_cut) / max(
            d.cheek_w_at_cut, template.cheek_w_at_cut)
        if cheek_pct > cheek_w_pct_max:
            continue
        if abs(d.yaw_deg - template.yaw_deg) > yaw_diff_max:
            continue
        # Gender HARD match (enforce only if BOTH predictions available)
        if (require_gender_match and template.predicted_gender != "?"
                and d.predicted_gender != "?"
                and template.predicted_gender != d.predicted_gender):
            continue
        # Age diff soft match (skip if either prediction missing)
        if (template.predicted_age > 0 and d.predicted_age > 0
                and abs(template.predicted_age - d.predicted_age) > age_diff_max):
            continue
        # Cut-to-chin percent diff (user-requested 10%) — avoids "short chin
        # vs long chin" mismatch that truncates the lower face after pasting
        ct_pct = abs(d.cut_to_chin_ratio - template.cut_to_chin_ratio) / max(
            d.cut_to_chin_ratio, template.cut_to_chin_ratio, 1e-6)
        if ct_pct > cut_to_chin_pct_max:
            continue
        pool.append(d)
    if len(pool) < k:
        return pool
    # Farthest-point sampling on dlib embeddings (greedy)
    embs = np.stack([d.emb for d in pool])
    # Start with the one closest to template (still illusion-eligible)
    dists_to_T = np.linalg.norm(embs - template.emb, axis=1)
    picks = [int(np.argmin(dists_to_T))]
    while len(picks) < k:
        # max-min distance to current pick set
        d_to_picks = np.min(
            np.stack([np.linalg.norm(embs - embs[p], axis=1) for p in picks]),
            axis=0,
        )
        d_to_picks[picks] = -1  # exclude already picked
        picks.append(int(np.argmax(d_to_picks)))
    return [pool[i] for i in picks]


# === Phase B: composite construction ==================================

def build_face_oval_mask(mp_lm: np.ndarray, shape: tuple[int, int],
                         feather_px: int = 3) -> np.ndarray:
    """Build a face oval mask from MediaPipe's 36 FACE_OVAL landmarks.

    The MediaPipe FACE_OVAL traces the actual face contour learned from
    millions of labeled faces — far more precise than dlib's 17 jaw points
    plus an interpolated forehead arc. Handles slight yaw / face-shape
    variation correctly so the mask hugs the real face boundary, not the
    background.

    Args:
      mp_lm: (478, 2) MediaPipe Face Mesh landmarks (pixel coords)
      shape: (h, w)
      feather_px: Gaussian feather radius on the boundary
    Returns:
      uint8 mask (0..255). Inside face = 255, outside = 0. Boundary feathered.
    """
    h, w = shape
    oval = mp_lm[MP_FACE_OVAL_IDX].astype(np.int32)
    # MP_FACE_OVAL_IDX is ordered clockwise around the face; use fillPoly
    # NOT convexHull — face shape is convex enough that hull is fine but the
    # ordered polygon preserves any nonconvexity (e.g. very pointed chin).
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [oval], 255)
    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0),
                                sigmaX=max(0.5, feather_px / 2.0),
                                sigmaY=max(0.5, feather_px / 2.0))
    return mask


def cheek_x_at_y(mp_lm: np.ndarray, y: float
                 ) -> tuple[float, float] | None:
    """Find the face's left and right boundary x-coordinates at a given y.

    Uses MP_BOTTOM_OVAL_IDX (the lower-half subset of FACE_OVAL) plus the
    top-half face contour above y_cut. Interpolates piecewise-linearly
    along the contour to find where horizontal line y intersects the
    face oval.

    Returns (left_x, right_x), or None if y is outside the face's y range.
    """
    oval = mp_lm[MP_FACE_OVAL_IDX]
    ys = oval[:, 1]
    if y < ys.min() or y > ys.max():
        return None
    # Find all segments that cross y
    crossings: list[float] = []
    n = len(oval)
    for i in range(n):
        j = (i + 1) % n
        y0, y1 = oval[i, 1], oval[j, 1]
        x0, x1 = oval[i, 0], oval[j, 0]
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            if y0 == y1:
                continue
            t = (y - y0) / (y1 - y0)
            x = x0 + t * (x1 - x0)
            crossings.append(float(x))
    if len(crossings) < 2:
        return None
    crossings.sort()
    return crossings[0], crossings[-1]   # leftmost and rightmost x at y


def cheek_align_warp_bottom(donor_rgb: np.ndarray, donor_mp_lm: np.ndarray,
                            template_mp_lm: np.ndarray,
                            y_cut: int) -> np.ndarray:
    """Per-row horizontal warp of donor's bottom half so the donor's face
    left/right boundary at each row y matches the template's face left/right
    boundary at that row.

    Above y_cut: pixels unchanged (template's top half will overwrite anyway).
    Below y_cut: linearly remap x so [left_donor, right_donor] -> [left_T, right_T].
    Outside donor's face boundary at that row: identity (no warp).

    This stretches/compresses donor's cheek-to-cheek width per row to match
    template's, without deforming donor's facial features vertically.
    """
    h, w = donor_rgb.shape[:2]
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    # Build per-row x-warp for y in [y_cut, h)
    for y in range(y_cut, h):
        d_bounds = cheek_x_at_y(donor_mp_lm, float(y))
        t_bounds = cheek_x_at_y(template_mp_lm, float(y))
        if d_bounds is None or t_bounds is None:
            continue
        lx_d, rx_d = d_bounds
        lx_t, rx_t = t_bounds
        if rx_d - lx_d < 1 or rx_t - lx_t < 1:
            continue
        # Inverse map: for destination row y, where in source should each x come from?
        # cv2.remap takes (map_x[y, x], map_y[y, x]) = "for output pixel (x, y), sample from source at that position".
        # We want output pixel x in [lx_t, rx_t] to receive source pixel at
        # x_src = lx_d + (x - lx_t) / (rx_t - lx_t) * (rx_d - lx_d)
        # Outside [lx_t, rx_t]: keep source x = output x.
        scale = (rx_d - lx_d) / (rx_t - lx_t)
        # Vectorised over the row
        xs = np.arange(w, dtype=np.float32)
        in_face = (xs >= lx_t) & (xs <= rx_t)
        new_x = np.where(in_face, lx_d + (xs - lx_t) * scale, xs)
        map_x[y, :] = new_x
    warped = cv2.remap(donor_rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REFLECT_101)
    return warped


def tps_apply_to_points(pts: np.ndarray, lm_src: np.ndarray,
                        lm_dst: np.ndarray) -> np.ndarray:
    """Approximate the TPS warp's effect on arbitrary points by computing,
    for each input pt, the displacement of its NEAREST landmark and applying
    that displacement. Sufficient for landmark coordinates that already lie
    near the TPS control points (dlib 68 are a subset of the face area
    covered by MediaPipe 478 anchors).

    For exact TPS evaluation we would re-build the TPS transform and call
    applyTransformation, but OpenCV's `ThinPlateSplineShapeTransformer` does
    not support that directly; nearest-neighbor displacement is the
    standard workaround for sparse follow-up landmarks.
    """
    out = pts.copy().astype(np.float32)
    for i, p in enumerate(pts):
        nn = int(np.argmin(np.linalg.norm(lm_src - p, axis=1)))
        disp = lm_dst[nn] - lm_src[nn]
        out[i] = p + disp
    return out


def align_donor_to_template_explicit(
        donor_rgb: np.ndarray, donor_lm: np.ndarray, donor_mp_lm: np.ndarray,
        template: IdAttr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Similarity-transform donor's rgb + landmarks → template frame.

    Same as align_donor_to_template but takes the donor's rgb and landmarks
    as explicit arguments (so we can pass the FRONTALIZED donor instead of
    the original donor.rgb / donor.lm / donor.mp_lm).
    """
    h_t, w_t = template.rgb.shape[:2]
    src_le = donor_lm[LEFT_EYE].mean(axis=0)
    src_re = donor_lm[RIGHT_EYE].mean(axis=0)
    dst_le = template.lm[LEFT_EYE].mean(axis=0)
    dst_re = template.lm[RIGHT_EYE].mean(axis=0)
    src_pts = np.float32([src_le, src_re])
    dst_pts = np.float32([dst_le, dst_re])
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        M = np.eye(2, 3, dtype=np.float32)
    warped = cv2.warpAffine(donor_rgb, M, (w_t, h_t),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT_101)

    def _apply_affine(pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
        h_ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        return (mat @ np.concatenate([pts.astype(np.float32), h_ones], axis=1).T).T.astype(np.float32)
    warped_lm = _apply_affine(donor_lm, M)
    warped_mp_lm = _apply_affine(donor_mp_lm, M)
    return warped, warped_lm, warped_mp_lm


def align_donor_to_template(donor: IdAttr, template: IdAttr
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Align donor.rgb to template's coordinate frame via a 2-point
    similarity transform on EYE CENTERS (rotation + isotropic scale +
    translation). This places donor's face at the same scale and yaw as
    template, without deforming donor's facial geometry — donor's nose,
    mouth, jaw shape, chin remain donor's identity-distinguishing features.

    Returns (warped_rgb, warped_lm) where warped_lm is donor.lm pushed
    through the same similarity matrix.

    Why similarity (not TPS):
      TPS warp with bottom-half landmarks pinned to template's positions
      forces donor's chin/mouth/jaw shape to match template — which erases
      the donor identity that's the whole point of composite illusion.
      Similarity transform preserves donor's facial shape, only normalizing
      scale + tilt + position. Jaw-width mismatch at the seam is handled by
      (a) donor-filter constraint |bottom_w_ratio Δ| ≤ 0.07 and
      (b) thin-strip Poisson blend at the cut line.
    """
    h_t, w_t = template.rgb.shape[:2]
    h_d, w_d = donor.rgb.shape[:2]
    src_le = donor.lm[LEFT_EYE].mean(axis=0)
    src_re = donor.lm[RIGHT_EYE].mean(axis=0)
    dst_le = template.lm[LEFT_EYE].mean(axis=0)
    dst_re = template.lm[RIGHT_EYE].mean(axis=0)
    src_pts = np.float32([src_le, src_re])
    dst_pts = np.float32([dst_le, dst_re])
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        M = np.eye(2, 3, dtype=np.float32)
    warped = cv2.warpAffine(donor.rgb, M, (w_t, h_t),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT_101)
    # Warp landmarks (both dlib 68 and MediaPipe 478)
    def _apply_affine(pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
        h_ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        return (mat @ np.concatenate([pts.astype(np.float32), h_ones], axis=1).T).T.astype(np.float32)
    warped_lm = _apply_affine(donor.lm, M)
    warped_mp_lm = _apply_affine(donor.mp_lm, M)
    return warped, warped_lm, warped_mp_lm


def _reinhard_ring_LAB(donor_bot_rgb: np.ndarray, T_ring_rgb: np.ndarray,
                       a_b_blend: float = 0.5) -> np.ndarray:
    """Reinhard LAB transfer on donor bottom half. L channel: full transfer
    (mean+std match T_ring). a, b channels: partial blend (a_b_blend
    fraction) — preserves donor's chroma identity (lip color, etc.).

    T_ring_rgb: pixels from a ring around the cut line on the template side
    (anchor for color statistics).
    """
    src_lab = cv2.cvtColor(donor_bot_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(T_ring_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    s = src_lab.reshape(-1, 3)
    r = ref_lab.reshape(-1, 3)
    s_mean = s.mean(0); s_std = s.std(0)
    r_mean = r.mean(0); r_std = r.std(0)
    # L full transfer
    out_L = (src_lab[..., 0] - s_mean[0]) / max(s_std[0], 1e-6)
    out_L = out_L * r_std[0] + r_mean[0]
    # a / b partial blend toward target
    out_a = src_lab[..., 1] + a_b_blend * (r_mean[1] - s_mean[1])
    out_b = src_lab[..., 2] + a_b_blend * (r_mean[2] - s_mean[2])
    out = np.stack([out_L, out_a, out_b], axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def make_composite(template: IdAttr, donor: IdAttr,
                   seam_band: int = 14,
                   ring_px: int = 24) -> CompositeResult:
    """Build all 4 composite conditions for (template, donor).

    Pipeline:
      1. Similarity-transform donor → template frame (eye-center anchored)
      2. Raw composite: T top + D_aligned bottom (paste at y_cut)
      3. L-channel ring transfer on D_aligned (preserves donor chroma id)
      4. Thin-strip seamlessClone NORMAL — mask is ONLY ±seam_band around
         y_cut, not the full bottom (avoids erasing donor's interior gradients)
      5. Compose face oval mask from T's TOP half + D_aligned's BOTTOM half
         landmarks (each side conforms to its own face shape)
      6. Apply mask: pixels outside → mid-gray
      7. V1 = aligned upright; V2 = bottom shifted dx; V3/V4 = inverted
    """
    T_rgb = template.rgb
    h, w = T_rgb.shape[:2]
    y_cut = int(round(template.y_cut))
    face_w = template.face_w
    mis_dx = int(round(0.5 * face_w))

    # 1. Similarity transform donor → template's frame (eye-anchored).
    #    No frontalization — TPS-based mirror-symmetrize was too distorting
    #    on lower face. Instead we filter at the donor-pick stage for
    #    (a) mouth centrality and (b) cut-line cheek-width matching.
    D_aligned, D_aligned_lm, D_aligned_mp_lm = align_donor_to_template_explicit(
        donor.rgb, donor.lm, donor.mp_lm, template)

    # 2. Per-row horizontal warp: donor cheek-x → template cheek-x at each
    #    y in [y_cut, h]. Eliminates residual "split face" artifact for the
    #    small width differences allowed past the 10% cheek-width filter.
    D_cheekwarped = cheek_align_warp_bottom(
        D_aligned, D_aligned_mp_lm, template.mp_lm, y_cut)

    # 2. Build raw composite canvas (T top + D_cheekwarped bottom)
    raw = T_rgb.copy()
    raw[y_cut:] = D_cheekwarped[y_cut:]

    # 3. L-channel ring color transfer on the cheek-warped donor bottom.
    #    Preserves donor chroma identity (lip color, etc.) while shifting
    #    luminance so the cut-line skin tone matches.
    ring_above = T_rgb[max(0, y_cut - ring_px):y_cut, :]
    ring_below_donor = D_cheekwarped[y_cut:min(h, y_cut + ring_px), :]
    T_lab = cv2.cvtColor(ring_above, cv2.COLOR_RGB2LAB).astype(np.float32)
    D_lab = cv2.cvtColor(ring_below_donor, cv2.COLOR_RGB2LAB).astype(np.float32)
    L_shift = float(T_lab[..., 0].mean() - D_lab[..., 0].mean())
    D_lab_full = cv2.cvtColor(D_cheekwarped, cv2.COLOR_RGB2LAB).astype(np.float32)
    D_lab_full[y_cut:, :, 0] = np.clip(D_lab_full[y_cut:, :, 0] + L_shift, 0, 255)
    D_aligned_recoloured = cv2.cvtColor(D_lab_full.astype(np.uint8),
                                        cv2.COLOR_LAB2RGB)
    canvas_pre_poisson = T_rgb.copy()
    canvas_pre_poisson[y_cut:] = D_aligned_recoloured[y_cut:]

    # 4. Poisson NORMAL_CLONE: paste donor's bottom into template's
    #    coordinate frame, smoothing the boundary at y_cut. NORMAL_CLONE
    #    (not MIXED — see [[project_illusionbench_eeg_pivot]]) preserves
    #    donor's gradients (= donor's identity) and only adjusts boundary
    #    pixels to match template at the seam.
    #
    #    src = D_aligned_recoloured (entire donor image after L-shift)
    #    dst = T_rgb (template)
    #    mask = bottom half (from y_cut down), eroded a few px from image
    #           edges to avoid OpenCV's "boundary touches image edge" error
    src_for_poisson = D_aligned_recoloured
    bot_mask = np.zeros((h, w), dtype=np.uint8)
    bot_mask[y_cut:h - 4, 4:w - 4] = 255
    bot_mask = cv2.erode(bot_mask, np.ones((3, 3), np.uint8), iterations=1)
    centre = (w // 2, (y_cut + h) // 2)
    try:
        v1_raw = cv2.seamlessClone(
            src_for_poisson, T_rgb.copy(), bot_mask, centre,
            cv2.NORMAL_CLONE)
    except cv2.error:
        # Fallback: linear alpha-blend across a ±seam_band strip
        v1_raw = canvas_pre_poisson.copy()
        for y in range(max(0, y_cut - seam_band), min(h, y_cut + seam_band)):
            t = (y - (y_cut - seam_band)) / max(1, 2 * seam_band)
            v1_raw[y] = (T_rgb[y].astype(np.float32) * (1 - t)
                         + canvas_pre_poisson[y].astype(np.float32) * t
                         ).astype(np.uint8)

    # 5. Compose face oval mask from MediaPipe 36-point FACE_OVAL.
    #    Because we've cheek-warped the donor's bottom to template's
    #    contour, BOTH faces share the same boundary at y >= y_cut → we
    #    can use ONE precise oval mask from template's MP landmarks for
    #    the whole composite (no top/bottom seam in the mask itself).
    oval_T = build_face_oval_mask(template.mp_lm, (h, w), feather_px=3)
    combined_oval = oval_T

    # 6. Apply face-oval mask
    v1 = _apply_oval_mask(v1_raw, combined_oval)

    # 7. Build V2 misaligned upright (uses same precise oval for top + shifted oval for bottom)
    v2 = _build_misaligned_v2(T_rgb, D_aligned_recoloured,
                              oval_T, oval_T, y_cut, mis_dx)

    # 8. V3 / V4: inverted
    v3 = np.ascontiguousarray(np.rot90(v1, 2))
    v4 = np.ascontiguousarray(np.rot90(v2, 2))

    # 9. QC metrics
    ssim_v1v2 = _ssim_in_face(v1, v2, combined_oval, y_cut, h)
    seam_disc = _seam_discontinuity(v1_raw, y_cut, band=3)
    lab_delta_cut = _lab_delta_at_cut(T_rgb, v1_raw, y_cut, ring_px)

    r = CompositeResult(
        template_name=template.name, donor_name=donor.name,
        y_cut=template.y_cut, face_w=face_w, mis_dx_px=float(mis_dx),
        ssim_v1v2=ssim_v1v2, seam_disc=seam_disc,
        lab_delta_cut=lab_delta_cut,
        v1=v1, v2=v2, v3=v3, v4=v4,
    )
    # Stash extra QC fields as attributes (saved into json)
    r.template_yaw = template.yaw_deg
    r.donor_yaw = donor.yaw_deg
    return r


def _apply_oval_mask(rgb: np.ndarray, oval_mask: np.ndarray) -> np.ndarray:
    """Composite rgb onto mid-gray background via the soft oval mask."""
    alpha = (oval_mask.astype(np.float32) / 255.0)[..., None]
    bg = np.full_like(rgb, BG_GRAY)
    out = (rgb.astype(np.float32) * alpha
           + bg.astype(np.float32) * (1.0 - alpha))
    return out.astype(np.uint8)


def _build_misaligned_v2(T_rgb: np.ndarray,
                          D_aligned_recoloured: np.ndarray,
                          oval_T: np.ndarray, oval_D: np.ndarray,
                          y_cut: int, dx: int) -> np.ndarray:
    """Build misaligned upright (V2): top half is template (unchanged from V1
    top); bottom half = D's recoloured aligned bottom, SHIFTED horizontally
    by dx pixels. No Poisson — the discontinuity is the manipulation.

    Mask = oval_T for top region + shifted oval_D for bottom region.
    """
    h, w = T_rgb.shape[:2]
    canvas = T_rgb.copy()
    bot_h = h - y_cut
    shifted_bot = np.full((bot_h, w, 3), BG_GRAY, dtype=np.uint8)
    if dx >= 0:
        shifted_bot[:, dx:w] = D_aligned_recoloured[y_cut:, :w - dx]
    else:
        s = -dx
        shifted_bot[:, :w - s] = D_aligned_recoloured[y_cut:, s:]
    canvas[y_cut:] = shifted_bot
    # Shifted oval for the bottom (so the mid-gray BG outside donor's face
    # doesn't get rendered as part of the face)
    shifted_oval_D = np.zeros_like(oval_D)
    if dx >= 0:
        shifted_oval_D[y_cut:, dx:w] = oval_D[y_cut:, :w - dx]
    else:
        s = -dx
        shifted_oval_D[y_cut:, :w - s] = oval_D[y_cut:, s:]
    # Composite: top half from T's oval, bottom half from shifted D's oval
    combined = np.zeros((h, w), dtype=np.uint8)
    combined[:y_cut, :] = oval_T[:y_cut, :]
    combined[y_cut:, :] = shifted_oval_D[y_cut:, :]
    return _apply_oval_mask(canvas, combined)


def _ssim_in_face(a: np.ndarray, b: np.ndarray,
                  v1_mask: np.ndarray,
                  y_cut: int, h: int,
                  bg_gray: int = BG_GRAY) -> float:
    """SSIM between V1 and V2 restricted to a region that's NON-mid-gray in
    EITHER image (so the metric reflects face-content differences, not
    matching mid-gray BG).

    For composite V1 vs V2: in V1 the bottom-half face content is at its
    canonical position; in V2 it's shifted by dx. SSIM should be low
    because the bottom content changed position. Restricting to where
    EITHER image has face content avoids the inflation we saw with the
    union-of-ovals mask (where most pixels are mid-gray in both → SSIM ≈ 1).
    """
    if v1_mask.sum() == 0:
        return 0.0
    # Build "is-face" boolean mask: pixel is face if it differs from BG gray
    # in either image (tolerance for feathered alpha)
    is_face_a = (np.abs(a.astype(np.int16) - bg_gray).max(axis=-1) > 12)
    is_face_b = (np.abs(b.astype(np.int16) - bg_gray).max(axis=-1) > 12)
    face_either = is_face_a | is_face_b
    if not face_either.any():
        return 0.0
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    s, S = ssim(ag, bg, full=True, data_range=255)
    return float(S[face_either].mean())


def _seam_discontinuity(img: np.ndarray, y_cut: int, band: int = 3) -> float:
    """Mean absolute vertical gradient in a horizontal band centered on y_cut.
    Lower = smoother seam. Reported in 0-255 luminance units."""
    h = img.shape[0]
    y0 = max(0, y_cut - band); y1 = min(h, y_cut + band)
    crop = cv2.cvtColor(img[y0:y1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    if crop.shape[0] < 2:
        return 0.0
    dy = np.abs(np.diff(crop, axis=0))
    return float(dy.mean())


def _lab_delta_at_cut(T_rgb: np.ndarray, composite_rgb: np.ndarray,
                      y_cut: int, ring_px: int) -> float:
    """|mean LAB of T-side ring| vs |mean LAB of composite-bottom-side ring|.
    Reports total Euclidean distance in LAB space. Lower = better blend."""
    h = T_rgb.shape[0]
    T_ring = T_rgb[max(0, y_cut - ring_px):y_cut, :]
    D_ring = composite_rgb[y_cut:min(h, y_cut + ring_px), :]
    T_lab = cv2.cvtColor(T_ring, cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(0)
    D_lab = cv2.cvtColor(D_ring, cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(0)
    return float(np.linalg.norm(T_lab - D_lab))


# === Phase C: QC ======================================================

def make_contact_4grid(result: CompositeResult, tile: int = 384) -> np.ndarray:
    """Build a 2x2 grid (V1 V2 / V3 V4) with labels. Returns RGB ndarray."""
    label_h = 28
    grid = np.full((2 * (tile + label_h), 2 * tile, 3), 255, dtype=np.uint8)
    panels = [(result.v1, "V1 aligned upright"),
              (result.v2, "V2 misaligned upright"),
              (result.v3, "V3 aligned inverted"),
              (result.v4, "V4 misaligned inverted")]
    for i, (im, label) in enumerate(panels):
        r, c = i // 2, i % 2
        y = r * (tile + label_h)
        x = c * tile
        small = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
        grid[y:y + tile, x:x + tile] = small
        cv2.putText(grid, label, (x + 6, y + tile + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Header line
    header = np.full((30, 2 * tile, 3), 245, dtype=np.uint8)
    cv2.putText(header,
                f"T={result.template_name}  D={result.donor_name}  "
                f"y_cut={result.y_cut:.0f}  face_w={result.face_w:.0f}  "
                f"SSIM={result.ssim_v1v2:.2f}  seam={result.seam_disc:.2f}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([header, grid], axis=0)


def save_result(r: CompositeResult, out_dir: Path) -> dict:
    """Save 4 PNGs + 1 JSON for one (T, D) pair, plus the 2x2 contact PNG."""
    base = f"{r.template_name}__d{r.donor_name}"
    for vk, im in [("v1", r.v1), ("v2", r.v2), ("v3", r.v3), ("v4", r.v4)]:
        Image.fromarray(im).save(out_dir / f"{base}.{vk}.png", optimize=True)
    meta = {
        "template": r.template_name, "donor": r.donor_name,
        "y_cut": r.y_cut, "face_w": r.face_w, "mis_dx_px": r.mis_dx_px,
        "ssim_v1v2": r.ssim_v1v2, "seam_disc": r.seam_disc,
        "lab_delta_cut": r.lab_delta_cut,
        "template_yaw_deg": getattr(r, "template_yaw", float("nan")),
        "donor_yaw_deg": getattr(r, "donor_yaw", float("nan")),
    }
    (out_dir / f"{base}.json").write_text(json.dumps(meta, indent=2))
    # Contact sheet
    contact = make_contact_4grid(r)
    Image.fromarray(contact).save(out_dir / f"{base}.contact.png", optimize=True)
    return meta


# === Main ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default="data/local_qc_partwhole",
                    help="Dir containing ffhq_*/{landmarks.json,thatcher_V1.png}")
    ap.add_argument("--output_dir", default="data/composite_pilot")
    ap.add_argument("--predictor", default="models/shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--recognizer", default="models/dlib_face_recognition_resnet_model_v1.dat")
    ap.add_argument("--n_templates", type=int, default=8,
                    help="Number of templates to generate composites for")
    ap.add_argument("--donors_per_template", type=int, default=3)
    ap.add_argument("--limit_loaded", type=int, default=0,
                    help="If >0, load at most this many identities (debug)")
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--no_insight", action="store_true",
                    help="Skip InsightFace gender/age prediction (faster but no gender hard-match)")
    args = ap.parse_args()

    src = Path(args.source_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] source: {src}")
    print(f"[pilot] output: {out}")

    embedder = FaceEmbedder(args.predictor, args.recognizer)
    insight = None
    if not args.no_insight:
        print("[pilot] loading InsightFace (genderage) …")
        insight = InsightAttrExtractor()
    t0 = time.monotonic()
    attrs = load_identities(src, embedder, limit=args.limit_loaded, insight=insight)
    print(f"[pilot] phase A: loaded {len(attrs)} identities  "
          f"({time.monotonic() - t0:.1f}s)")
    if len(attrs) < args.donors_per_template + 1:
        print("[pilot] too few identities for any template; abort")
        return

    # Pick templates: shuffle and take first n_templates that yield ≥ K donors
    import random
    rng = random.Random(args.seed)
    candidates = list(attrs)
    rng.shuffle(candidates)
    template_keep: list[IdAttr] = []
    donor_map: dict[str, list[IdAttr]] = {}
    for cand in candidates:
        donors = pick_donors(cand, attrs, k=args.donors_per_template)
        if len(donors) >= args.donors_per_template:
            template_keep.append(cand)
            donor_map[cand.name] = donors
        if len(template_keep) >= args.n_templates:
            break
    print(f"[pilot] phase B: picked {len(template_keep)} templates with "
          f"≥{args.donors_per_template} donors each")
    if not template_keep:
        print("[pilot] no qualifying template — relax donor filter and rerun")
        return

    manifest_rows: list[dict] = []
    contact_sheets: list[np.ndarray] = []
    t1 = time.monotonic()
    for ti, T in enumerate(template_keep):
        donors = donor_map[T.name]
        print(f"  [{ti+1}/{len(template_keep)}] T={T.name}  "
              f"donors=[{', '.join(d.name for d in donors)}]")
        for D in donors:
            try:
                r = make_composite(T, D)
            except Exception as e:
                print(f"    !! failed {T.name} × {D.name}: {e!r}")
                continue
            meta = save_result(r, out)
            manifest_rows.append(meta)
            contact_sheets.append(make_contact_4grid(r, tile=256))
    print(f"[pilot] phase B done: {len(manifest_rows)} (T,D) pairs in "
          f"{(time.monotonic() - t1):.1f}s")

    # Write global pilot manifest
    if manifest_rows:
        mpath = out / "manifest.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader(); w.writerows(manifest_rows)
        print(f"[pilot] manifest: {mpath}")

    # Stack contact sheets into one big QC sheet (4 per row, tile width 512)
    if contact_sheets:
        per_row = 2
        rows = []
        for i in range(0, len(contact_sheets), per_row):
            row_imgs = contact_sheets[i:i + per_row]
            max_h = max(im.shape[0] for im in row_imgs)
            padded = []
            for im in row_imgs:
                if im.shape[0] < max_h:
                    pad = np.full((max_h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
                    im = np.concatenate([im, pad], axis=0)
                padded.append(im)
            row_concat = np.concatenate(padded, axis=1)
            rows.append(row_concat)
        max_w = max(r.shape[1] for r in rows)
        padded_rows = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded_rows.append(r)
        big = np.concatenate(padded_rows, axis=0)
        Image.fromarray(big).save(out / "QC_contact_sheet.png", optimize=True)
        print(f"[pilot] QC contact sheet: {out / 'QC_contact_sheet.png'}")

    # Quick stats
    if manifest_rows:
        ssim_vals = [r["ssim_v1v2"] for r in manifest_rows]
        seam_vals = [r["seam_disc"] for r in manifest_rows]
        lab_vals  = [r["lab_delta_cut"] for r in manifest_rows]
        print()
        print(f"[pilot] QC summary ({len(manifest_rows)} pairs)")
        print(f"  SSIM(V1, V2) in oval  : mean={np.mean(ssim_vals):.3f}  "
              f"min={np.min(ssim_vals):.3f}  max={np.max(ssim_vals):.3f}")
        print(f"    target [0.55, 0.80] — too high = misalign invisible, "
              f"too low = TPS warp blew up")
        print(f"  seam_disc (luminance) : mean={np.mean(seam_vals):.2f}  "
              f"max={np.max(seam_vals):.2f}  target < 8")
        print(f"  lab_delta at cut     : mean={np.mean(lab_vals):.2f}  "
              f"max={np.max(lab_vals):.2f}  target < 5")


if __name__ == "__main__":
    main()
