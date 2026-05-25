"""stimuli/composite_partwhole_heuristic.py — composite + part-whole transforms.

Path C (hybrid) Part 2: composite-face and part-whole stimulus generators
that operate on cached source images (e.g., GPT-generated V1 portraits).
Uses heuristic positions (no MediaPipe), with luminance-band matching at
seam boundaries to handle diverse source lighting.

COMPOSITE (Young 1987 variant):
  V1 aligned_same      = top(i) + bot(i)            (== rgb_i)
  V2 aligned_diff      = top(i) + bot(j)            (foil bottom, aligned)
  V3 misaligned_same   = top(i) + bot(i, shifted)
  V4 misaligned_diff   = top(i) + bot(j, shifted)
  CSI = mean d(V1, V2) / mean d(V3, V4)

PART-WHOLE (Tanaka & Sengco 1997 variant):
  V1 whole_target      = rgb_i
  V2 whole_foil        = rgb_i with j's eye region pasted in
  V3 part_target       = i's eye region on neutral 128-gray canvas
  V4 part_foil         = j's eye region on the SAME canvas/bbox position
  PWI = mean d(V1, V2) / mean d(V3, V4)
  (Tanaka direction: PWI > 1 means whole-context discriminates better.)

Seam handling:
  - composite: 8 px Gaussian alpha fade along the horizontal cut, +
    histogram luminance matching of bottom-half luminance to top-half
    border (matches median V of HSV).
  - part-whole: ellipse-shaped soft alpha + the same hist-match.

Pixel-baseline (computed in the project's compute_metrics.py):
  composite raw baseline ≈ 1.10 (top + foil-bottom changes < shifted variant).
  part-whole raw baseline ≈ 1.20 (gray canvas is mostly constant).
The existing pipeline divides by these.

Pair selection:
  Deterministic non-self shuffle by (i + 1 + seed_offset) mod N — same
  scheme as generate_composite_ffhq.py + generate_partwhole_ffhq.py for
  consistency with the project's existing conventions.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def luminance_match(bottom: np.ndarray, top_border: np.ndarray) -> np.ndarray:
    """Shift bottom's HSV-V channel so its median matches top_border's median.
    Soft, prevents 'two-person stitching' visible luminance jump at the seam.
    """
    bot_hsv = cv2.cvtColor(bottom, cv2.COLOR_RGB2HSV).astype(np.float32)
    top_hsv = cv2.cvtColor(top_border, cv2.COLOR_RGB2HSV).astype(np.float32)
    delta = float(np.median(top_hsv[..., 2]) - np.median(bot_hsv[..., 2]))
    bot_hsv[..., 2] = np.clip(bot_hsv[..., 2] + delta, 0, 255)
    return cv2.cvtColor(bot_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def soft_seam_blend(canvas: np.ndarray, cut_y: int, half_band: int):
    """Apply a Gaussian alpha fade across a +/-half_band strip centered on cut_y.
    In-place modifies `canvas` to blur the discontinuity at the seam."""
    h, w = canvas.shape[:2]
    y0 = max(0, cut_y - half_band); y1 = min(h, cut_y + half_band)
    if y1 <= y0 + 1:
        return
    strip = canvas[y0:y1].copy()
    blurred = cv2.GaussianBlur(strip, (0, 0),
                                sigmaX=max(1, half_band / 3),
                                sigmaY=max(1, half_band / 3))
    alpha = np.zeros((y1 - y0, 1, 1), dtype=np.float32)
    center = cut_y - y0
    for r in range(y1 - y0):
        d = abs(r - center)
        alpha[r, 0, 0] = max(0, 1 - d / half_band)
    canvas[y0:y1] = (strip.astype(np.float32) * (1 - alpha) +
                     blurred.astype(np.float32) * alpha).astype(np.uint8)


def compose(top_i: np.ndarray, bot_src: np.ndarray, cut_y: int,
             shift_px: int, h: int, w: int, seam_band: int,
             do_lum_match: bool) -> np.ndarray:
    """Compose top of identity i with given bottom source, with optional shift
    and luminance matching."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:cut_y] = top_i
    bot_h, bot_w = bot_src.shape[:2]
    target = np.zeros((bot_h, w, 3), dtype=np.uint8)
    if shift_px == 0:
        target[:, :bot_w] = bot_src
    elif shift_px > 0:
        end = min(w, bot_w + shift_px)
        target[:, shift_px:end] = bot_src[:, :end - shift_px]
    else:
        s = -shift_px; end = min(w, bot_w - s)
        target[:, :end] = bot_src[:, s:s + end]
    canvas[cut_y:cut_y + bot_h] = target
    if do_lum_match and cut_y > 4:
        # Match bottom luminance to a thin strip just above the seam
        border_top = top_i[max(0, cut_y - 4):cut_y]
        bot_region = canvas[cut_y:cut_y + bot_h]
        canvas[cut_y:cut_y + bot_h] = luminance_match(bot_region, border_top)
    soft_seam_blend(canvas, cut_y, seam_band)
    return canvas


def make_composite(rgb_i: np.ndarray, rgb_j: np.ndarray,
                    cut_frac: float, shift_px: int, seam_band: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb_i.shape[:2]
    cut_y = int(h * cut_frac)
    top_i = rgb_i[:cut_y]; bot_i = rgb_i[cut_y:]
    bot_j = rgb_j[cut_y:] if rgb_j.shape[0] >= h else cv2.resize(rgb_j, (w, h))[cut_y:]
    v1 = compose(top_i, bot_i, cut_y, 0,        h, w, seam_band, do_lum_match=False)
    v2 = compose(top_i, bot_j, cut_y, 0,        h, w, seam_band, do_lum_match=True)
    v3 = compose(top_i, bot_i, cut_y, shift_px, h, w, seam_band, do_lum_match=False)
    v4 = compose(top_i, bot_j, cut_y, shift_px, h, w, seam_band, do_lum_match=True)
    return v1, v2, v3, v4


def ellipse_alpha(h: int, w: int, cx: float, cy: float,
                   a: float, b: float, feather: float) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(a), int(b)),
                0, 0, 360, 255, thickness=-1)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather, sigmaY=feather)
    return (m.astype(np.float32) / 255.0)[..., None]


def make_partwhole(rgb_i: np.ndarray, rgb_j: np.ndarray, feather: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """V1 = i; V2 = i with j's eye region pasted; V3 = i's eye region on gray;
    V4 = j's eye region on gray (at the SAME bbox position)."""
    h, w = rgb_i.shape[:2]
    # Eye+brow region: combined ellipse spanning both eyes at ~36/64% width,
    # ~40% height. Combined bounding ellipse roughly: cx=50%, cy=40%, a=22%, b=8%.
    cx, cy = 0.50 * w, 0.40 * h
    a, b = 0.22 * w, 0.08 * h
    alpha = ellipse_alpha(h, w, cx, cy, a, b, feather)

    eyes_i_region = rgb_i.astype(np.float32) * alpha     # for V3 (just for shape)
    eyes_j_region = rgb_j.astype(np.float32) * alpha

    # V2: i with j's eye region pasted in. Luminance-match j's eye region first.
    # Take a thin annulus around the ellipse from rgb_i as reference luminance.
    annulus_a = a * 1.4; annulus_b = b * 1.4
    annulus = ellipse_alpha(h, w, cx, cy, annulus_a, annulus_b, 2.0) - alpha
    annulus = np.clip(annulus, 0, 1)
    # Sample i's skin around the ellipse for luminance reference
    skin_ref = (rgb_i.astype(np.float32) * annulus)
    skin_ref_mask = (annulus[..., 0] > 0.2)
    if skin_ref_mask.any():
        skin_ref_px = rgb_i[skin_ref_mask]
        # j eyes block (the region we'll paste)
        eye_block_j = (rgb_j.astype(np.float32) * alpha).astype(np.uint8)
        eye_block_lum_matched = luminance_match(eye_block_j, skin_ref_px[None, :, :])
        # Compose
        v2 = (rgb_i.astype(np.float32) * (1 - alpha) +
              eye_block_lum_matched.astype(np.float32) * alpha).astype(np.uint8)
    else:
        v2 = (rgb_i.astype(np.float32) * (1 - alpha) +
              eyes_j_region).astype(np.uint8)

    # V3 / V4 on gray canvas
    gray = np.full_like(rgb_i, 128)
    v3 = (gray.astype(np.float32) * (1 - alpha) +
          rgb_i.astype(np.float32) * alpha).astype(np.uint8)
    v4 = (gray.astype(np.float32) * (1 - alpha) +
          rgb_j.astype(np.float32) * alpha).astype(np.uint8)
    v1 = rgb_i.copy()
    return v1, v2, v3, v4


def make_contact_sheet(images, titles, out_path, tile_size=384):
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
    n = len(sources)
    if n < 2:
        raise SystemExit("need ≥ 2 sources for paired paradigms")
    print(f"[init] {n} sources for composite + part-whole")

    rng = np.random.default_rng(args.seed)
    pair_idx = (np.arange(n) + 1 + rng.integers(1, n - 1)) % n
    for i in range(n):
        if pair_idx[i] == i:
            pair_idx[i] = (i + 1) % n

    manifest_rows = []
    contact_sheets = {"composite": [], "partwhole": []}

    rgbs = [np.array(Image.open(p).convert("RGB")) for p in sources]
    for i in range(n):
        j = int(pair_idx[i])
        rgb_i, rgb_j = rgbs[i], rgbs[j]
        gid = sources[i].parent.name
        gdir = out_root / gid
        gdir.mkdir(parents=True, exist_ok=True)

        # Composite
        cv1, cv2, cv3, cv4 = make_composite(rgb_i, rgb_j,
                                              cut_frac=args.cut_frac,
                                              shift_px=args.shift_px,
                                              seam_band=args.seam_band)
        for cond, im in [("V1_aligned_same", cv1),
                          ("V2_aligned_diff", cv2),
                          ("V3_misaligned_same", cv3),
                          ("V4_misaligned_diff", cv4)]:
            p = gdir / f"composite_{cond}.png"
            Image.fromarray(im).save(p)
            manifest_rows.append({"group_id": gid, "paradigm": "composite",
                                   "condition": cond, "paired_with": j,
                                   "path": str(p)})
        cs = gdir / "composite_contact.png"
        make_contact_sheet([cv1, cv2, cv3, cv4],
                           ["V1 aligned_same", "V2 aligned_diff",
                            "V3 misaligned_same", "V4 misaligned_diff"], cs)
        contact_sheets["composite"].append(cs)

        # Part-Whole
        pv1, pv2, pv3, pv4 = make_partwhole(rgb_i, rgb_j, feather=args.pw_feather)
        for cond, im in [("V1_whole_target", pv1),
                          ("V2_whole_foil", pv2),
                          ("V3_part_target", pv3),
                          ("V4_part_foil", pv4)]:
            p = gdir / f"partwhole_{cond}.png"
            Image.fromarray(im).save(p)
            manifest_rows.append({"group_id": gid, "paradigm": "partwhole",
                                   "condition": cond, "paired_with": j,
                                   "path": str(p)})
        cs = gdir / "partwhole_contact.png"
        make_contact_sheet([pv1, pv2, pv3, pv4],
                           ["V1 whole_target", "V2 whole_foil",
                            "V3 part_target", "V4 part_foil"], cs)
        contact_sheets["partwhole"].append(cs)

        print(f"  {gid} (paired with {sources[j].parent.name})")

    # Combined QC: 2 groups for each paradigm
    for para, sheets in contact_sheets.items():
        if len(sheets) >= 2:
            rows = [np.array(Image.open(p).convert("RGB")) for p in sheets[:2]]
            max_w = max(r.shape[1] for r in rows)
            padded = []
            for r in rows:
                if r.shape[1] < max_w:
                    pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                    r = np.concatenate([r, pad], axis=1)
                padded.append(r)
            Image.fromarray(np.concatenate(padded, axis=0)).save(
                out_root / f"qc_2groups_{para}.png"
            )
            print(f"  combined QC for {para}: {out_root}/qc_2groups_{para}.png")

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader(); w.writerows(manifest_rows)
    print(f"[done] {n} groups × 2 paradigms × 4 conditions = {len(manifest_rows)} images")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",
                   default="data/stimuli_gpt_diverse_pilot",
                   help="dir with group_*/V1_source.png")
    p.add_argument("--output_dir",
                   default="data/stimuli_gpt_diverse_compositepw_heuristic")
    p.add_argument("--cut_frac", type=float, default=0.58,
                   help="horizontal cut as fraction of height (nose-tip ≈ 0.55-0.60)")
    p.add_argument("--shift_px", type=int, default=80,
                   help="bottom-half horizontal shift (px at 1024×1024)")
    p.add_argument("--seam_band", type=int, default=10,
                   help="seam blend half-band in px")
    p.add_argument("--pw_feather", type=float, default=12.0,
                   help="part-whole eye-ellipse feather sigma")
    p.add_argument("--seed", type=int, default=20260525)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
