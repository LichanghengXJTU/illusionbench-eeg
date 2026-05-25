"""stimuli/classical_cv/color_transfer.py — Reinhard 2001 LAB color transfer.

Reinhard, Ashikhmin, Gooch & Shirley (2001) "Color Transfer between Images."
IEEE Computer Graphics & Applications.

Pure statistical algorithm: matches mean + std of L*a*b* channels of a SOURCE
image (or set of source pixels) to a REFERENCE image (or pixel set). Standard
classical-CV baseline for cross-image color/luminance matching.
"""
from __future__ import annotations
import cv2
import numpy as np


def _stats(lab: np.ndarray):
    """Return per-channel (mean, std) for an HxWx3 or Nx3 LAB image/array."""
    flat = lab.reshape(-1, 3).astype(np.float32)
    return flat.mean(axis=0), flat.std(axis=0)


def reinhard_transfer(source_rgb: np.ndarray,
                       reference_rgb: np.ndarray) -> np.ndarray:
    """Recolor `source_rgb` so its LAB statistics match `reference_rgb`.
    Both inputs are uint8 RGB. Reference can be a full image OR a small
    pixel sample (any shape with .reshape(-1, 3) valid).
    Returns uint8 RGB."""
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) \
        if reference_rgb.ndim == 3 else reference_rgb.astype(np.float32)
    src_mean, src_std = _stats(src_lab)
    ref_mean, ref_std = _stats(ref_lab)
    # Standardize source, then re-scale to reference stats.
    out = (src_lab - src_mean) / np.maximum(src_std, 1e-6)
    out = out * ref_std + ref_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def reinhard_transfer_masked(source_rgb: np.ndarray, source_mask: np.ndarray,
                              reference_pixels_rgb: np.ndarray) -> np.ndarray:
    """Apply Reinhard transfer to source_rgb but use only pixels where
    source_mask > 0 to compute source stats. Returns uint8 RGB (same shape
    as source_rgb), with the transfer applied everywhere (or only inside
    mask if `mask_only=True` — but we apply globally for smoothness)."""
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_pixels_rgb.reshape(1, -1, 3),
                            cv2.COLOR_RGB2LAB).astype(np.float32)
    mask = source_mask > 0
    if not mask.any():
        return source_rgb
    src_pixels = src_lab[mask]
    s_mean = src_pixels.mean(axis=0); s_std = src_pixels.std(axis=0)
    r_flat = ref_lab.reshape(-1, 3)
    r_mean = r_flat.mean(axis=0); r_std = r_flat.std(axis=0)
    out = (src_lab - s_mean) / np.maximum(s_std, 1e-6)
    out = out * r_std + r_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
