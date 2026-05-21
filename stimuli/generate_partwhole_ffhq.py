"""stimuli/generate_partwhole_ffhq.py — Part-Whole illusion battery from FFHQ.

Implements a variant of the Tanaka & Sengco (1997) part-whole paradigm using
the previously-selected FFHQ identities + cached landmarks.

For each base identity i (with paired identity j, drawn from another id by
deterministic shuffle), we manipulate the EYE region (encompassing both eyes
+ brows, the most-load-bearing feature for face identity per Schyns 2002):

  V1 whole_target = i with i's own eyes (= original rgb_i)
  V2 whole_foil   = i with j's eyes pasted in over i's eye region
  V3 part_target  = i's eyes on neutral gray background (same canvas size)
  V4 part_foil    = j's eyes on neutral gray background (same canvas size, same position)

Part-Whole Index (PWI_pert) = mean d(V1, V2) / mean d(V3, V4)

Human prediction: PWI > 1 — whole-face context amplifies the discrimination of
feature swap (holistic binding); in isolation the eyes are more easily judged
identical. A model with no holistic binding gives PWI ≈ 1 (whole-face context
provides no boost).

Same V1/V2/V3/V4 naming convention as Thatcher / composite for downstream
script compatibility.
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stimuli.generate_thatcher import (
    paste_with_alpha,
    LEFT_EYE_REGION,
    RIGHT_EYE_REGION,
)


# Combined eye+brow indices — both eyes treated as one region for swap.
EYE_REGION = sorted(set(LEFT_EYE_REGION) | set(RIGHT_EYE_REGION))


def get_eyes_bbox(lm_arr: np.ndarray, w: int, h: int, padding: int):
    pts = lm_arr[EYE_REGION]
    x1 = max(0, int(pts[:, 0].min() - padding))
    y1 = max(0, int(pts[:, 1].min() - padding))
    x2 = min(w, int(pts[:, 0].max() + padding))
    y2 = min(h, int(pts[:, 1].max() + padding))
    return x1, y1, x2, y2


def crop_eyes(img: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()


def gray_canvas_with_patch(canvas_h: int, canvas_w: int, patch: np.ndarray,
                           paste_bbox, blur_sigma: float) -> np.ndarray:
    """Make a neutral-gray (128) canvas of size (canvas_h, canvas_w, 3) with the
    eye patch pasted at paste_bbox, soft-blended at the boundary."""
    canvas = np.full((canvas_h, canvas_w, 3), 128, dtype=np.uint8)
    x1, y1, x2, y2 = paste_bbox
    h_p, w_p = y2 - y1, x2 - x1
    # If the donor patch shape differs (different bbox sizes per identity),
    # resize to fit the target bbox (this is the standard handling)
    if patch.shape[:2] != (h_p, w_p):
        patch = cv2.resize(patch, (w_p, h_p), interpolation=cv2.INTER_CUBIC)
    paste_with_alpha(canvas, patch, paste_bbox, blur_sigma)
    return canvas


def swap_eyes_in_whole(img_target: np.ndarray, eyes_donor: np.ndarray,
                       paste_bbox, blur_sigma: float) -> np.ndarray:
    out = img_target.copy()
    x1, y1, x2, y2 = paste_bbox
    h_p, w_p = y2 - y1, x2 - x1
    if eyes_donor.shape[:2] != (h_p, w_p):
        eyes_donor = cv2.resize(eyes_donor, (w_p, h_p), interpolation=cv2.INTER_CUBIC)
    paste_with_alpha(out, eyes_donor, paste_bbox, blur_sigma)
    return out


@dataclass
class Identity:
    identity_id: int
    ffhq_name: str
    paired_with: int
    paired_ffhq_name: str
    canonical_png_path: str
    landmarks_json_path: str
    eye_bbox: str  # JSON


def main(args):
    src_root = Path(args.src_root)
    src_identity_manifest = src_root / "identity_manifest.csv"

    out_root = Path(args.output_root)
    stim_dir = out_root / "partwhole"
    identities_dir = out_root / "identities"
    landmarks_dir = out_root / "identity_landmarks"
    for d in (stim_dir, identities_dir, landmarks_dir):
        d.mkdir(parents=True, exist_ok=True)

    src_records = []
    with open(src_identity_manifest) as f:
        for row in csv.DictReader(f):
            src_records.append(row)
    print(f"[init] {len(src_records)} source identities")

    rng = np.random.default_rng(args.seed)

    n = len(src_records)
    pair_idx = (np.arange(n) + 1 + rng.integers(1, n - 1)) % n
    for i in range(n):
        if pair_idx[i] == i:
            pair_idx[i] = (i + 1) % n

    manifest_rows = []
    identity_records = []
    blur_sigma = args.blur_sigma
    target_n = args.target_n if args.target_n > 0 else n

    pbar = tqdm(range(min(n, target_n)), desc="partwhole")
    for k in pbar:
        i = k
        j = int(pair_idx[i])
        rec_i = src_records[i]; rec_j = src_records[j]
        identity_id = int(rec_i["identity_id"])
        paired_id = int(rec_j["identity_id"])

        png_i = Path(rec_i["canonical_png_path"])
        png_j = Path(rec_j["canonical_png_path"])
        lm_i = Path(rec_i["landmarks_json_path"])
        lm_j = Path(rec_j["landmarks_json_path"])
        if not all(p.exists() for p in (png_i, png_j, lm_i, lm_j)):
            continue
        bgr_i = cv2.imread(str(png_i)); rgb_i = cv2.cvtColor(bgr_i, cv2.COLOR_BGR2RGB)
        bgr_j = cv2.imread(str(png_j)); rgb_j = cv2.cvtColor(bgr_j, cv2.COLOR_BGR2RGB)
        h, w = rgb_i.shape[:2]
        if rgb_j.shape[:2] != (h, w):
            rgb_j = cv2.resize(rgb_j, (w, h), interpolation=cv2.INTER_CUBIC)

        with open(lm_i) as f: lm_data_i = json.load(f)
        with open(lm_j) as f: lm_data_j = json.load(f)
        lm_arr_i = np.array(lm_data_i["landmarks"], dtype=np.float32)
        lm_arr_j = np.array(lm_data_j["landmarks"], dtype=np.float32)

        eye_pad = lm_data_i.get("eye_pad_used", max(8, int(min(h, w) * 0.02)))
        # Use bigger padding for the part-whole transfer so we cover both eyes + some brow + lid
        eye_pad = max(eye_pad, int(min(h, w) * 0.035))  # ~36 px at 1024
        bbox_i = get_eyes_bbox(lm_arr_i, w, h, eye_pad)
        bbox_j = get_eyes_bbox(lm_arr_j, w, h, eye_pad)

        eyes_i = crop_eyes(rgb_i, bbox_i)
        eyes_j = crop_eyes(rgb_j, bbox_j)

        v1 = rgb_i.copy()                                                        # whole_target
        v2 = swap_eyes_in_whole(rgb_i, eyes_j, bbox_i, blur_sigma)               # whole_foil
        v3 = gray_canvas_with_patch(h, w, eyes_i, bbox_i, blur_sigma)            # part_target
        v4 = gray_canvas_with_patch(h, w, eyes_j, bbox_i, blur_sigma)            # part_foil

        ident_sym = identities_dir / f"identity_{identity_id:04d}.png"
        lm_sym = landmarks_dir / f"identity_{identity_id:04d}.json"
        if not ident_sym.exists():
            try: ident_sym.symlink_to(png_i.resolve())
            except FileExistsError: pass
        if not lm_sym.exists():
            try: lm_sym.symlink_to(lm_i.resolve())
            except FileExistsError: pass

        identity_records.append(Identity(
            identity_id=identity_id,
            ffhq_name=rec_i.get("lfw_name", rec_i.get("ffhq_name", "")),
            paired_with=paired_id,
            paired_ffhq_name=rec_j.get("lfw_name", rec_j.get("ffhq_name", "")),
            canonical_png_path=str(ident_sym),
            landmarks_json_path=str(lm_sym),
            eye_bbox=json.dumps([int(x) for x in bbox_i]),
        ))

        for vname, vimg in [
            ("V1_upright_normal",   v1),  # whole_target
            ("V2_upright_thatched", v2),  # whole_foil
            ("V3_inverted_normal",  v3),  # part_target
            ("V4_inverted_thatched",v4),  # part_foil
        ]:
            path = stim_dir / f"identity_{identity_id:04d}_{vname}.png"
            Image.fromarray(vimg).save(path)
            manifest_rows.append({
                "stim_id": f"thatcher_id{identity_id:04d}_{vname}",
                "identity_id": identity_id,
                "lfw_name": rec_i.get("lfw_name", rec_i.get("ffhq_name", "")),
                "paradigm": "partwhole",
                "condition": vname,
                "orientation": "whole" if vname.startswith(("V1","V2")) else "part",
                "thatcherized": int(vname.startswith(("V2","V4"))),
                "path": str(path),
            })

        pbar.set_postfix(accepted=len(identity_records))

    manifest_path = out_root / "thatcher_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    identity_manifest_path = out_root / "identity_manifest.csv"
    with open(identity_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(identity_records[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in identity_records])

    print(f"[done] {len(identity_records)} identities, {len(manifest_rows)} stimuli")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--target_n", type=int, default=0)
    p.add_argument("--blur_sigma", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
