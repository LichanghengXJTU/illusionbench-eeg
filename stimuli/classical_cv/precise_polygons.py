"""stimuli/classical_cv/precise_polygons.py — landmark-ordered polygons.

dlib's 68-point predictor gives landmarks in a canonical, ORDERED way around
each feature. Connecting them IN ORDER (not via convex hull) produces a
polygon that hugs the actual visible feature boundary much tighter than the
convex hull. This is what makes seamlessClone seams disappear: the mask
only edits the feature interior, not the surrounding skin.

Conventions (300-W 68-point, dlib subject-frame):
  Eye:  6 ordered points clockwise around the eye opening
        (LEFT_EYE = 36..41 starting at outer corner, going up around top,
         down to inner corner, around bottom, back to outer corner)
  Brow: 5 ordered points along the eyebrow upper edge (LEFT_BROW = 17..21)
  Mouth outer lip: 12 ordered points (48..59)
  Jaw: 17 ordered points (0..16) defining the lower face boundary

Combined eye+brow polygon:
  upper boundary = brow points (17..21 / 22..26)
  lower boundary = eye lower arc (41..36 / 47..42 reversed)
  joined into a closed polygon: brow_left → brow_right → eye_bottom → close
"""
from __future__ import annotations
import cv2
import numpy as np


# dlib 68-pt indices
JAW         = list(range(0, 17))
LEFT_BROW   = list(range(17, 22))   # 5 points, ordered outer→inner
RIGHT_BROW  = list(range(22, 27))   # 5 points, ordered inner→outer
NOSE_BRIDGE = list(range(27, 31))
NOSE_TIP    = 30
NOSE_BASE   = list(range(31, 36))
LEFT_EYE    = list(range(36, 42))   # 6 points, ordered around eye
RIGHT_EYE   = list(range(42, 48))   # 6 points
MOUTH_OUTER = list(range(48, 60))   # 12 points, outer lip clockwise
MOUTH_INNER = list(range(60, 68))   # 8 points


def eye_brow_polygon(landmarks: np.ndarray, eye_idx: list[int],
                       brow_idx: list[int], pad_above_brow: float = 0.05,
                       iod: float = 100.0) -> np.ndarray:
    """Return ordered polygon (N, 1, 2) int32 for the FULL eye+brow region.
    Brow upper line (padded up) + eye lower arc, with the SIDES connecting
    them via the eye's outer + inner corners. Produces a tight shape that
    encloses the entire brow AND the entire eye opening (so seamlessClone
    sees the iris/pupil as 'feature interior').

    Approach: take the CONVEX HULL of {padded_brow_points ∪ eye_points}.
    This guarantees the polygon encloses everything we care about while
    still tightly hugging the visible feature boundary. The convex-hull
    result here is much tighter than just bbox because brow + eye points
    are inherently arc-shaped, not rectangular."""
    brow_pts = landmarks[brow_idx].astype(np.float32).copy()
    pad_px = pad_above_brow * iod
    brow_pts[:, 1] -= pad_px      # shift brow upward to include brow hair
    eye_pts = landmarks[eye_idx].astype(np.float32)
    all_pts = np.concatenate([brow_pts, eye_pts], axis=0).astype(np.int32)
    hull = cv2.convexHull(all_pts)
    return hull


def mouth_polygon(landmarks: np.ndarray) -> np.ndarray:
    """Return ordered polygon for the OUTER mouth boundary (12 points)."""
    return landmarks[MOUTH_OUTER].astype(np.int32).reshape(-1, 1, 2)


def face_oval_polygon(landmarks: np.ndarray, iod: float,
                       forehead_pad_frac: float = 0.50) -> np.ndarray:
    """Return ordered polygon enclosing the visible face (jaw + estimated
    forehead). Used to clip composite swaps to within the actual face area,
    preventing background pixels from being recolored."""
    jaw_pts = landmarks[JAW].astype(np.float32)
    # Forehead estimate: take brow midline + pad upward by forehead_pad_frac × IOD
    brow_top = landmarks[LEFT_BROW + RIGHT_BROW].astype(np.float32)
    forehead_y = float(brow_top[:, 1].mean() - forehead_pad_frac * iod)
    # Build top arc of forehead: at jaw extreme-left and extreme-right x, at forehead_y
    fl = np.array([jaw_pts[0, 0], forehead_y], dtype=np.float32)
    fr = np.array([jaw_pts[-1, 0], forehead_y], dtype=np.float32)
    # Add a few midline forehead points so the arc is smooth
    midxs = np.linspace(jaw_pts[0, 0], jaw_pts[-1, 0], 7)[1:-1]
    forehead_arc = np.stack([midxs, np.full_like(midxs, forehead_y)], axis=1)
    # Order: jaw (L→R) → forehead arc (R→L) → close
    forehead_full = np.concatenate(
        [fr[None, :], forehead_arc[::-1], fl[None, :]], axis=0
    )
    poly = np.concatenate([jaw_pts, forehead_full], axis=0)
    return poly.astype(np.int32).reshape(-1, 1, 2)


def polygon_mask(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fill an ordered polygon into a uint8 binary mask."""
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def polygon_bbox(poly: np.ndarray, shape: tuple[int, int],
                  pad: int = 4) -> tuple[int, int, int, int]:
    h, w = shape
    xs = poly[:, 0, 0]; ys = poly[:, 0, 1]
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + pad), min(h, int(ys.max()) + pad))
