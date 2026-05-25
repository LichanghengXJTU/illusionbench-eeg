"""stimuli/classical_cv/feature_warp.py — within-identity TPS feature warp.

Replaces the cross-identity feature swap. Take the subject's own dlib
landmarks, shift a small subset (the feature being warped), build a
TPS transform that smoothly remaps source → target. The unchanged
landmarks (jaw, brows, other features) act as ANCHORS that pin those
regions to identity, so the warp is LOCAL to the targeted feature.

Two warp DIRECTIONS per feature ("a" and "b") — bidirectional warps
double the data and provide cleaner PWI baseline (foil distribution).

PW conformance: warp manipulation isolates SHAPE-based holistic
processing without confounding skin tone / ethnicity / identity cues
that cross-identity feature transfer introduces. See methods.

Warp parameters (deterministic, ratified 2026-05-26 with user):
  Nose:
    direction a (longer + narrower): tip down 10% IOD, alars inward 5%
    direction b (shorter + wider):   tip up 10% IOD, alars outward 5%
  Eyes:
    direction a (open wider):  upper lids up 6%, lower lids down 4%
    direction b (more closed): upper lids down 4%, lower lids up 3%
  Mouth:
    direction a (widen): corners outward 10% IOD
    direction b (purse): corners inward 5% IOD
"""
from __future__ import annotations
import cv2
import numpy as np


# === Landmark shift specs ===================================================

def warp_landmarks(lm: np.ndarray, feature: str, direction: str,
                   iod: float) -> np.ndarray:
    """Return new (68, 2) landmark array with feature-specific points shifted.

    Other landmarks are returned UNCHANGED — they act as anchors for TPS
    so the warp is localized to the feature region."""
    out = lm.copy()
    if feature == "nose":
        if direction == "a":     # longer + narrower
            out[30, 1] += 0.10 * iod        # nose tip down
            out[28, 1] += 0.03 * iod        # bridge mid down slightly
            out[29, 1] += 0.06 * iod        # bridge low down
            out[31, 0] += 0.04 * iod        # left alar inward
            out[35, 0] -= 0.04 * iod        # right alar inward
            out[32, 0] += 0.02 * iod
            out[34, 0] -= 0.02 * iod
        elif direction == "b":   # shorter + wider
            out[30, 1] -= 0.08 * iod        # nose tip up
            out[28, 1] -= 0.02 * iod
            out[29, 1] -= 0.05 * iod
            out[31, 0] -= 0.05 * iod        # left alar outward
            out[35, 0] += 0.05 * iod
            out[32, 0] -= 0.03 * iod
            out[34, 0] += 0.03 * iod
        else:
            raise ValueError(f"unknown nose direction: {direction}")
    elif feature == "eye":
        if direction == "a":     # open wider
            for idx in [37, 38]:                # left eye upper lid
                out[idx, 1] -= 0.06 * iod
            for idx in [43, 44]:                # right eye upper lid
                out[idx, 1] -= 0.06 * iod
            for idx in [40, 41]:                # left eye lower lid
                out[idx, 1] += 0.04 * iod
            for idx in [46, 47]:                # right eye lower lid
                out[idx, 1] += 0.04 * iod
        elif direction == "b":   # narrow / more closed
            for idx in [37, 38, 43, 44]:
                out[idx, 1] += 0.04 * iod        # upper lids down
            for idx in [40, 41, 46, 47]:
                out[idx, 1] -= 0.03 * iod        # lower lids up
        else:
            raise ValueError(f"unknown eye direction: {direction}")
    elif feature == "mouth":
        if direction == "a":     # widen
            out[48, 0] -= 0.10 * iod        # left corner out
            out[54, 0] += 0.10 * iod        # right corner out
            out[60, 0] -= 0.06 * iod        # inner corners follow
            out[64, 0] += 0.06 * iod
        elif direction == "b":   # purse / narrow
            out[48, 0] += 0.05 * iod        # corners inward
            out[54, 0] -= 0.05 * iod
            out[60, 0] += 0.03 * iod
            out[64, 0] -= 0.03 * iod
        else:
            raise ValueError(f"unknown mouth direction: {direction}")
    else:
        raise ValueError(f"unknown feature: {feature}")
    return out


# === TPS warp ===============================================================

def _image_corner_anchors(h: int, w: int) -> np.ndarray:
    """8 anchor points around image perimeter that don't move. Pins TPS
    extrapolation to identity outside the face landmark region (otherwise
    background pixels get unbounded warps)."""
    return np.array([
        [0,       0      ],
        [w - 1,   0      ],
        [0,       h - 1  ],
        [w - 1,   h - 1  ],
        [w // 2,  0      ],
        [w // 2,  h - 1  ],
        [0,       h // 2 ],
        [w - 1,   h // 2 ],
    ], dtype=np.float32)


def tps_warp_image(rgb: np.ndarray, lm_src: np.ndarray,
                   lm_dst: np.ndarray, border_value: int = 0) -> np.ndarray:
    """Apply TPS warp to rgb so pixel content at lm_src positions ends up
    at lm_dst positions. Unchanged landmarks (where lm_dst == lm_src) act
    as anchors that pin those regions to identity.

    Image-perimeter anchor points are ADDED to both source and target shape
    (unmoved) so the TPS extrapolation outside the face is constrained —
    without this, background/hair pixels get wavy distortion artifacts.

    OpenCV convention:
      tps.estimateTransformation(transformingShape, targetShape, matches)
      tps.warpImage(transformingImage)
    The transform maps `transformingShape` -> `targetShape`. We want the
    image warped so lm_src positions are remapped to lm_dst.
    Therefore: transformingShape = lm_src, targetShape = lm_dst.
    """
    h, w = rgb.shape[:2]
    anchors = _image_corner_anchors(h, w)
    src_all = np.concatenate([lm_src.astype(np.float32), anchors], axis=0)
    dst_all = np.concatenate([lm_dst.astype(np.float32), anchors], axis=0)
    tps = cv2.createThinPlateSplineShapeTransformer(regularizationParameter=0.0)
    src = src_all.reshape(1, -1, 2)
    dst = dst_all.reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_all))]
    tps.estimateTransformation(dst, src, matches)
    warped = tps.warpImage(rgb)
    return warped


# === Verify warp magnitudes are within ecological range =====================

def warp_distance_summary(lm_src: np.ndarray, lm_dst: np.ndarray,
                          iod: float) -> dict:
    """Return summary stats on per-point warp magnitudes (px / IOD).
    Useful for QC and for paper methods table.

    Healthy range: max shift ~5-15% of IOD; mean shift ~1-5% of IOD."""
    diffs = lm_dst - lm_src
    mags = np.linalg.norm(diffs, axis=1)
    moved_mask = mags > 0.5    # only points that actually moved
    return {
        "n_moved": int(moved_mask.sum()),
        "max_shift_iod": float(mags.max() / iod) if mags.max() > 0 else 0.0,
        "mean_shift_moved_iod": (float(mags[moved_mask].mean() / iod)
                                  if moved_mask.any() else 0.0),
    }
