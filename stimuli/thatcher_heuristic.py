"""stimuli/thatcher_heuristic.py — heuristic-ellipse Thatcher transform.

Path C (hybrid) proof-of-concept. No MediaPipe dependency (avoids the
server's protobuf/tf compat issue). Uses heuristic ellipse masks at
standard portrait face proportions, which work for centered face
photographs at 1024×1024 (matches all our GPT V1 sources).

For each V1 source:
  - Extract left-eye + right-eye + mouth elliptical regions (heuristic positions)
  - Rotate each region 180° in place
  - Alpha-blend back using the ellipse as a soft (Gaussian-feathered) mask
  - V1 = original, V2 = thatcherized, V3 = rotate-180(V1), V4 = rotate-180(V2)

Pixel-baseline ISI sanity: V3 = rot180(V1) and V4 = rot180(V2) exactly,
so d_pixel(V1,V2) = d_pixel(V3,V4) — ISI_pixel = 1.000 by construction.

Used as a local complement to API-generated V1 sources (path C in the
2026-05-25 architecture pivot).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_ellipse_mask(h: int, w: int, cx: float, cy: float,
                        a: float, b: float, feather_px: float) -> np.ndarray:
    """Soft ellipse mask in (h, w) float32 in [0, 1].
    Center (cx, cy) and semi-axes (a, b) in pixels."""
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(a), int(b)),
                0, 0, 360, 255, thickness=-1)
    if feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather_px, sigmaY=feather_px)
    return m.astype(np.float32) / 255.0


def rotate_region_inplace(img: np.ndarray, cx: float, cy: float,
                           a: float, b: float, feather_px: float) -> np.ndarray:
    """Return a new image: like img, but the elliptical region around (cx, cy)
    with semi-axes (a, b) is 180° rotated in place, alpha-blended at the
    boundary with `feather_px` Gaussian sigma."""
    h, w = img.shape[:2]
    # bbox enclosing the ellipse (with margin for safe rotation)
    margin = int(np.ceil(feather_px * 3)) + 2
    x1 = max(0, int(cx - a) - margin); x2 = min(w, int(cx + a) + margin)
    y1 = max(0, int(cy - b) - margin); y2 = min(h, int(cy + b) + margin)
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return img.copy()

    # rotate the bbox content 180° (= flip both axes)
    patch = img[y1:y2, x1:x2].copy()
    rotated = cv2.rotate(patch, cv2.ROTATE_180)

    # local mask in bbox coords. The ellipse is centered at (cx-x1, cy-y1).
    local_cx, local_cy = cx - x1, cy - y1
    local_h, local_w = y2 - y1, x2 - x1
    mask = build_ellipse_mask(local_h, local_w, local_cx, local_cy,
                               a, b, feather_px)
    # the rotated-mask is the 180° rotation of the original mask
    mask_rot = cv2.rotate(mask, cv2.ROTATE_180)
    # effective mask = max(mask, mask_rot) so the blend boundary is symmetric.
    eff = np.maximum(mask, mask_rot)[..., None]

    base = img[y1:y2, x1:x2].astype(np.float32)
    new = rotated.astype(np.float32)
    blended = base * (1.0 - eff) + new * eff
    result = img.copy()
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def thatcherize_heuristic(img_rgb: np.ndarray, feather_px: float = 6.0
                           ) -> np.ndarray:
    """Apply heuristic Thatcher (eye+mouth 180° in-place) to centered portrait."""
    h, w = img_rgb.shape[:2]
    # eye centers at ~(36% w, 40% h) and ~(64% w, 40% h); eye size ~10% w × 6% h.
    # mouth at ~(50% w, 70% h); size ~14% w × 7% h.
    regions = [
        ("L_eye",  0.36 * w, 0.40 * h, 0.10 * w, 0.06 * h),
        ("R_eye",  0.64 * w, 0.40 * h, 0.10 * w, 0.06 * h),
        ("mouth",  0.50 * w, 0.70 * h, 0.14 * w, 0.07 * h),
    ]
    out = img_rgb.copy()
    for _name, cx, cy, a, b in regions:
        out = rotate_region_inplace(out, cx, cy, a, b, feather_px)
    return out


def rotate_180(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_180)


def make_contact_sheet(images: list[np.ndarray], titles: list[str],
                        out_path: Path, tile_size: int = 384):
    """Make a horizontal contact sheet of N tiles with titles below each."""
    n = len(images)
    title_strip = 30
    sheet = np.full((tile_size + title_strip, tile_size * n, 3), 255, dtype=np.uint8)
    for i, (img, title) in enumerate(zip(images, titles)):
        resized = cv2.resize(img, (tile_size, tile_size),
                              interpolation=cv2.INTER_AREA)
        sheet[:tile_size, i*tile_size:(i+1)*tile_size] = resized
        cv2.putText(sheet, title,
                    (i*tile_size + 10, tile_size + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(sheet).save(out_path)


def main(args):
    src_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sources = sorted(src_root.glob("group_*/V1_source.png"))
    if not sources:
        raise SystemExit(f"no group_*/V1_source.png under {src_root}")
    print(f"[init] {len(sources)} source images")

    manifest_rows = []
    contact_paths = []
    for src_path in sources:
        gid = src_path.parent.name  # e.g. "group_00"
        rgb = np.array(Image.open(src_path).convert("RGB"))
        v1 = rgb.copy()
        v2 = thatcherize_heuristic(rgb, feather_px=args.feather_px)
        v3 = rotate_180(v1)
        v4 = rotate_180(v2)
        gdir = out_root / gid
        gdir.mkdir(parents=True, exist_ok=True)
        for cond, im in [("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)]:
            p = gdir / f"thatcher_{cond}.png"
            Image.fromarray(im).save(p)
            manifest_rows.append({
                "group_id": gid, "condition": cond,
                "paradigm": "thatcher_heuristic", "path": str(p),
            })
        # contact sheet for this group
        cs_path = gdir / "thatcher_contact.png"
        make_contact_sheet([v1, v2, v3, v4],
                           ["V1 upright_normal", "V2 upright_thatched",
                            "V3 inverted_normal", "V4 inverted_thatched"],
                           cs_path)
        contact_paths.append(cs_path)
        print(f"  {gid}: V1-V4 saved + contact sheet")

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader(); w.writerows(manifest_rows)

    # Combined contact sheet: 3 groups vertically (for quick QC)
    if len(contact_paths) >= 3:
        rows = []
        for p in contact_paths[:3]:
            rows.append(np.array(Image.open(p).convert("RGB")))
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        combined = np.concatenate(padded, axis=0)
        Image.fromarray(combined).save(out_root / "qc_3groups_4tiles.png")
        print(f"  combined QC sheet: {out_root}/qc_3groups_4tiles.png")

    print(f"[done] {len(sources)} groups × 4 conditions = {len(manifest_rows)} images")
    print(f"  manifest: {out_root}/manifest.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",
                   default="data/stimuli_gpt_diverse_pilot",
                   help="dir containing group_*/V1_source.png")
    p.add_argument("--output_dir",
                   default="data/stimuli_gpt_diverse_thatcher_heuristic")
    p.add_argument("--feather_px", type=float, default=6.0,
                   help="Gaussian feather for ellipse boundaries")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
