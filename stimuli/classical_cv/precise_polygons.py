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
                       brow_idx: list[int],
                       pad_above_brow: float = 0.06,
                       pad_below_eye: float = 0.13,
                       pad_lateral:    float = 0.07,
                       iod: float = 100.0) -> np.ndarray:
    """Return convex hull enclosing the FULL eye region:
       - brow (padded UP by pad_above_brow × IOD: brow hair)
       - eye opening (the 6-point arc)
       - eye bag region (eye lower points padded DOWN by pad_below_eye × IOD)
       - lateral makeup region (all points pushed outward from polygon centroid
         by pad_lateral × IOD: captures eyeshadow / outer-canthus crow's-feet
         / inner-corner makeup)
    Defaults tuned for typical FFHQ portraits at 1024×1024.
    """
    brow_pts = landmarks[brow_idx].astype(np.float32).copy()
    eye_pts = landmarks[eye_idx].astype(np.float32).copy()
    above_px = pad_above_brow * iod
    below_px = pad_below_eye * iod
    lateral_px = pad_lateral * iod
    # 1. Pad brow upward to include brow hair (above the landmark line)
    brow_pts[:, 1] -= above_px
    # 2. Pad LOWER eye landmarks downward to include eye bags / under-eye area
    #    (only the lower-half eye points: those with y >= eye centroid)
    eye_cy_native = eye_pts[:, 1].mean()
    is_lower = eye_pts[:, 1] >= eye_cy_native
    eye_pts[is_lower, 1] += below_px
    # 3. Pad all points outward from polygon centroid (lateral expansion)
    all_pts = np.concatenate([brow_pts, eye_pts], axis=0)
    cx = float(all_pts[:, 0].mean())
    cy = float(all_pts[:, 1].mean())
    for i in range(len(all_pts)):
        dx, dy = all_pts[i, 0] - cx, all_pts[i, 1] - cy
        norm = float(np.hypot(dx, dy))
        if norm > 1e-3:
            all_pts[i, 0] += (dx / norm) * lateral_px
            all_pts[i, 1] += (dy / norm) * lateral_px
    hull = cv2.convexHull(all_pts.astype(np.int32))
    return hull


def mouth_polygon(landmarks: np.ndarray, pad_frac: float = 0.07,
                    iod: float = 100.0) -> np.ndarray:
    """12-point outer-lip polygon, dilated outward by pad_frac × IOD to
    include the immediate lip-line / philtrum / chin-top transition zone
    (so seamlessClone has skin context to blend against)."""
    pts = landmarks[MOUTH_OUTER].astype(np.float32)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    pad_px = pad_frac * iod
    out = pts.copy()
    for i in range(len(out)):
        dx, dy = out[i, 0] - cx, out[i, 1] - cy
        n = float(np.hypot(dx, dy))
        if n > 1e-3:
            out[i, 0] += (dx / n) * pad_px
            out[i, 1] += (dy / n) * pad_px
    return out.astype(np.int32).reshape(-1, 1, 2)


def polygon_centroid(poly: np.ndarray) -> tuple[int, int]:
    """Vertex-averaged centroid (good approximation for our small polygons)."""
    pts = poly.reshape(-1, 2).astype(np.float32)
    return int(round(pts[:, 0].mean())), int(round(pts[:, 1].mean()))


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
