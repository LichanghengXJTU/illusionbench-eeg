"""stimuli/classical_cv/composite_classical.py — Young 1987 composite via classical CV.

Per Young, Hellawell & Hay (1987):
  - Same-sex paired identities (we use same demographic cluster)
  - Horizontal cut at the nose tip (landmark 33)
  - Aligned: top + bottom directly stacked
  - Misaligned: bottom shifted ~60 px (Young's original number)

Pipeline:
  - dlib 68-pt landmarks for both i, j (already cached from Phase 2)
  - Rigid affine align j → i frame using bilateral eye anchors (the two
    eye centers act as a 2-point similarity transform: scale + rotate +
    translate). This puts j's face in i's coordinate frame so the nose-tip
    cut line is consistent.
  - Reinhard LAB color transfer: j_aligned → match i's skin tone (sampled
    from cheek annulus).
  - Cut at i's nose-tip y. Top of i + bottom of j_aligned_recolored.
  - cv2.seamlessClone (NORMAL_CLONE) the bottom-half patch into i's
    canvas with a strip mask across the cut → Poisson-invisible seam.
  - V3/V4: same as V1/V2 but bottom shifted +60 px.

Pixel baseline ISI (see analysis/compute_metrics.py): ≈ 1.10 raw for
composite, ≈ 1.0 after pixel-correction.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stimuli.classical_cv.color_transfer import (
    reinhard_transfer, reinhard_transfer_masked,
)


NOSE_TIP_IDX = 30
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))


def rigid_align(src_rgb: np.ndarray, src_lm: np.ndarray,
                  dst_lm: np.ndarray, out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Align src image to dst's coordinate frame using the two eye centers
    as anchors (a similarity transform: scale + rotate + translate).
    Returns (warped_image, warped_landmarks)."""
    src_le = src_lm[LEFT_EYE_IDX].mean(axis=0)
    src_re = src_lm[RIGHT_EYE_IDX].mean(axis=0)
    dst_le = dst_lm[LEFT_EYE_IDX].mean(axis=0)
    dst_re = dst_lm[RIGHT_EYE_IDX].mean(axis=0)
    # Compute similarity transform
    src_pts = np.float32([src_le, src_re])
    dst_pts = np.float32([dst_le, dst_re])
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        # Fallback: identity
        M = np.eye(2, 3, dtype=np.float32)
    h, w = out_size
    warped = cv2.warpAffine(src_rgb, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    # Warp landmarks too
    lm_h = np.concatenate([src_lm, np.ones((68, 1))], axis=1)
    lm_warped = (M @ lm_h.T).T
    return warped, lm_warped.astype(np.float32)


def cheek_skin_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Binary mask of cheek skin area (for sampling reference color)."""
    h, w = shape
    # Cheek roughly = polygon of {jaw 1-15 } above nose tip, EXCLUDING eye+brow+mouth
    jaw = landmarks[1:16].astype(np.int32)
    nose_tip_y = int(landmarks[NOSE_TIP_IDX, 1])
    # Use lower-jaw region between just under eyes and above mouth
    eye_y = int((landmarks[36:48, 1]).mean())
    mouth_y = int((landmarks[48:68, 1]).min())
    # Trapezoidal cheek = jaw polygon clipped between eye_y + offset and mouth_y - offset
    poly = jaw.copy()
    poly = np.array([(p[0], min(max(p[1], eye_y + 10), mouth_y - 10)) for p in poly],
                     dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(poly), 255)
    return mask


def make_composite(rgb_i: np.ndarray, lm_i: np.ndarray,
                    rgb_j: np.ndarray, lm_j: np.ndarray,
                    shift_px: int = 60, seam_band: int = 12
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb_i.shape[:2]
    # Align j to i's frame
    j_aligned, lm_j_aligned = rigid_align(rgb_j, lm_j, lm_i, (h, w))
    # Reinhard color transfer: j_aligned → match i's cheek tone
    cheek_mask_i = cheek_skin_mask(lm_i, (h, w))
    cheek_pixels_i = rgb_i[cheek_mask_i > 0]
    if len(cheek_pixels_i) > 100:
        j_recolor = reinhard_transfer(j_aligned, cheek_pixels_i[None, :, :])
    else:
        j_recolor = j_aligned

    cut_y = int(lm_i[NOSE_TIP_IDX, 1])

    def _build(top_src: np.ndarray, bot_src: np.ndarray, shift: int) -> np.ndarray:
        canvas = top_src.copy()
        # bottom half from bot_src, optionally shifted right
        bot = bot_src[cut_y:].copy()
        bot_h = bot.shape[0]
        target = np.zeros((bot_h, w, 3), dtype=np.uint8)
        if shift == 0:
            target[:] = bot
        elif shift > 0:
            end = min(w, w + shift)
            target[:, shift:] = bot[:, :w - shift]
        else:
            s = -shift
            target[:, :w - s] = bot[:, s:]
        canvas[cut_y:cut_y + bot_h] = target
        # Build a thin horizontal strip mask centered on cut_y for Poisson blend
        strip = np.zeros((h, w), dtype=np.uint8)
        strip[max(0, cut_y - seam_band):min(h, cut_y + seam_band), :] = 255
        # Use seamlessClone of the WHOLE bottom-half region for color smoothing
        try:
            bot_only = np.zeros_like(canvas)
            bot_only[cut_y:cut_y + bot_h] = target
            mask_bot = np.zeros((h, w), dtype=np.uint8)
            mask_bot[cut_y:cut_y + bot_h, :] = 255
            # Erode the mask edges slightly to avoid seamlessClone boundary issues
            mask_bot = cv2.erode(mask_bot, np.ones((3, 3), np.uint8))
            center = (w // 2, cut_y + bot_h // 2)
            blended = cv2.seamlessClone(bot_only, top_src.copy(), mask_bot,
                                          center, cv2.NORMAL_CLONE)
            return blended
        except cv2.error:
            # Fallback: just Gaussian-fade the seam
            y0 = max(0, cut_y - seam_band); y1 = min(h, cut_y + seam_band)
            strip_img = canvas[y0:y1].copy()
            blurred = cv2.GaussianBlur(strip_img, (0, 0),
                                         sigmaX=max(1, seam_band / 3),
                                         sigmaY=max(1, seam_band / 3))
            alpha = np.zeros((y1 - y0, 1, 1), dtype=np.float32)
            ctr = cut_y - y0
            for r in range(y1 - y0):
                alpha[r, 0, 0] = max(0, 1 - abs(r - ctr) / seam_band)
            canvas[y0:y1] = (strip_img.astype(np.float32) * (1 - alpha) +
                              blurred.astype(np.float32) * alpha).astype(np.uint8)
            return canvas

    v1 = _build(rgb_i, rgb_i, shift=0)         # aligned_same  (same identity, no shift)
    v2 = _build(rgb_i, j_recolor, shift=0)     # aligned_diff  (top_i + bot_j_recolor)
    v3 = _build(rgb_i, rgb_i, shift=shift_px)  # misaligned_same
    v4 = _build(rgb_i, j_recolor, shift=shift_px)  # misaligned_diff
    return v1, v2, v3, v4


def make_contact_sheet(images, titles, out_path, tile=384):
    n = len(images)
    sheet = np.full((tile + 30, tile * n, 3), 255, dtype=np.uint8)
    for i, (im, t) in enumerate(zip(images, titles)):
        sheet[:tile, i*tile:(i+1)*tile] = cv2.resize(
            im, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, t, (i*tile + 10, tile + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(sheet).save(out_path)


def main(args):
    """Pair existing Phase-2 IDs via demographic clustering, generate composite."""
    from stimuli.classical_cv.demographic_cluster import (
        face_descriptor, cluster_identities, within_cluster_pairs,
    )

    src_root = Path(args.phase2_dir)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Load Phase-2 identities and their landmarks
    records = []
    for lm_path in sorted(src_root.glob("identity_*/landmarks.json")):
        meta = json.loads(lm_path.read_text())
        img_path = lm_path.parent / "thatcher_V1.png"
        rgb = np.array(Image.open(img_path).convert("RGB"))
        lm = np.array(meta["landmarks"], dtype=np.float32)
        bbox = tuple(meta["bbox"])
        descriptor = face_descriptor(rgb, lm, bbox)
        records.append({
            "identity_id": meta["identity_id"],
            "ffhq_name": meta["ffhq_name"],
            "rgb": rgb,
            "lm": lm,
            "bbox": bbox,
            "descriptor": descriptor,
        })
    n = len(records)
    print(f"[init] loaded {n} Phase-2 identities")

    cluster_map = cluster_identities(records, k=args.k_clusters)
    pairs = within_cluster_pairs(cluster_map)
    print(f"[cluster] k={args.k_clusters}; pair distribution:")
    for cid in sorted(set(cluster_map.values())):
        members = [r["identity_id"] for r in records if cluster_map[r["identity_id"]] == cid]
        print(f"  cluster {cid}: {len(members)} ids = {members}")

    manifest = []
    contact_sheets = []
    for r in records:
        i = r["identity_id"]
        j = pairs.get(i, i)
        if j == i:
            print(f"  [skip] id {i} singleton cluster")
            continue
        rec_j = next(rr for rr in records if rr["identity_id"] == j)
        v1, v2, v3, v4 = make_composite(
            r["rgb"], r["lm"], rec_j["rgb"], rec_j["lm"],
            shift_px=args.shift_px, seam_band=args.seam_band,
        )
        idir = out_root / f"identity_{i:04d}"
        idir.mkdir(parents=True, exist_ok=True)
        for cond, im in [("V1_aligned_same", v1), ("V2_aligned_diff", v2),
                          ("V3_misaligned_same", v3), ("V4_misaligned_diff", v4)]:
            p = idir / f"composite_{cond}.png"
            Image.fromarray(im).save(p)
            manifest.append({"identity_id": i, "paired_with": j,
                              "cluster": cluster_map[i],
                              "condition": cond, "path": str(p)})
        cs = idir / "contact.png"
        make_contact_sheet([v1, v2, v3, v4],
                           ["V1 aligned_same", "V2 aligned_diff",
                            "V3 misaligned_same", "V4 misaligned_diff"], cs)
        contact_sheets.append(cs)
        print(f"  [pair] id {i} (cluster {cluster_map[i]}) ↔ id {j}")

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    # Combined QC: 4 identity contact sheets stacked
    if len(contact_sheets) >= 4:
        rows = [np.array(Image.open(p).convert("RGB"))
                for p in contact_sheets[:4]]
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        Image.fromarray(np.concatenate(padded, axis=0)).save(
            out_root / "qc_4identities.png")
        print(f"  combined QC: {out_root}/qc_4identities.png")

    print(f"\n[done] {len(manifest)//4} composite identities × 4 conditions")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase2_dir", default="data/classical_cv_phase2_thatcher")
    p.add_argument("--output_dir", default="data/classical_cv_phase3_composite")
    p.add_argument("--k_clusters", type=int, default=5)
    p.add_argument("--shift_px", type=int, default=60)
    p.add_argument("--seam_band", type=int, default=12)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
