"""stimuli/generate_thatcher_v2.py — Thatcher with contour-mask rotation.

v2 of the Thatcher generator. Key change vs generate_thatcher_ffhq.py: instead
of rotating a rectangular bbox region and pasting back with a Gaussian-faded
rectangular alpha, we rotate the bbox region (the rotation itself is still
rectangular — geometrically required) but compose the rotated patch back using
a **polygon mask following the eye/eyebrow/lips landmark contour**, with a
narrow Gaussian fade at the polygon boundary.

This removes the visible rectangular paste-seams that the original generator
produces (confirmed by visual audit of qc/ffhq_contact.png 2026-05-24).

Mathematically: in the V1→V2 difference, only pixels INSIDE the polygon
change. Pixels outside the polygon (but inside the old bbox) are preserved
unchanged. The V3↔V4 pair has the same property (since global rotation
commutes with the polygon-masked rotation locally).

Pixel-baseline ISI sanity is preserved: V3 = rot180(V1), V4 = rot180(V2),
so d_pixel(V1,V2) = d_pixel(V3,V4) exactly.

Source images: can be either FFHQ identities or GPT-generated identities;
the script accepts either a `--ffhq_shard_dir` for FFHQ tar shards or a
`--gpt_image_dir` for pre-generated source PNGs.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import random
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

mp_fm = mp.solutions.face_mesh


def _connection_points(connections) -> list[int]:
    pts: set[int] = set()
    for a, b in connections:
        pts.add(a); pts.add(b)
    return sorted(pts)


LEFT_EYE_PTS = _connection_points(mp_fm.FACEMESH_LEFT_EYE)
RIGHT_EYE_PTS = _connection_points(mp_fm.FACEMESH_RIGHT_EYE)
LEFT_BROW_PTS = _connection_points(mp_fm.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW_PTS = _connection_points(mp_fm.FACEMESH_RIGHT_EYEBROW)
LEFT_EYE_REGION = sorted(set(LEFT_EYE_PTS) | set(LEFT_BROW_PTS))
RIGHT_EYE_REGION = sorted(set(RIGHT_EYE_PTS) | set(RIGHT_BROW_PTS))
LIPS_PTS = _connection_points(mp_fm.FACEMESH_LIPS)


@dataclass
class Identity:
    identity_id: int
    source_name: str
    canonical_png_path: str
    landmarks_json_path: str


def landmarks_to_array(landmarks, img_w: int, img_h: int) -> np.ndarray:
    return np.array([[lm.x * img_w, lm.y * img_h] for lm in landmarks], dtype=np.float32)


def region_bbox(landmark_arr: np.ndarray, indices: list[int],
                w: int, h: int, padding: int) -> tuple[int, int, int, int]:
    pts = landmark_arr[indices]
    x1 = max(0, int(pts[:, 0].min() - padding))
    y1 = max(0, int(pts[:, 1].min() - padding))
    x2 = min(w, int(pts[:, 0].max() + padding))
    y2 = min(h, int(pts[:, 1].max() + padding))
    return x1, y1, x2, y2


def contour_polygon_mask(landmark_arr: np.ndarray, indices: list[int],
                          bbox: tuple[int, int, int, int],
                          dilate_px: int, feather_sigma: float) -> np.ndarray:
    """Build a soft binary mask (h, w) at the bbox crop region, where the
    interior is the convex hull of the landmark points (in bbox-local
    coordinates), dilated outward by `dilate_px`, then feathered by
    `feather_sigma` Gaussian.

    Why convex hull rather than the FACEMESH connection polygon directly:
    the MP eye/lip connection sets are not always strictly convex polygons
    in pixel space; using cv2.convexHull on the landmark cluster yields a
    well-defined enclosing region that hugs the feature boundary smoothly.
    For Thatcher this is the correct fade region — we want everything
    inside the feature contour to flip, and a smooth fade at the boundary.
    """
    x1, y1, x2, y2 = bbox
    h_p, w_p = y2 - y1, x2 - x1
    if h_p <= 1 or w_p <= 1:
        return np.zeros((max(h_p, 1), max(w_p, 1)), dtype=np.float32)
    pts = landmark_arr[indices].copy()
    pts[:, 0] -= x1
    pts[:, 1] -= y1
    pts = pts.astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w_p - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_p - 1)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h_p, w_p), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    if feather_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0),
                                sigmaX=feather_sigma, sigmaY=feather_sigma)
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def thatcherize_contour(img_rgb: np.ndarray, landmark_arr: np.ndarray,
                         eye_pad: int, mouth_pad: int,
                         dilate_px: int, feather_sigma: float) -> np.ndarray:
    """For each of {left eye+brow, right eye+brow, lips}: extract the bbox,
    rotate 180°, paste back using the CONTOUR polygon mask (not the bbox)."""
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    for region_indices, pad in [
        (LEFT_EYE_REGION, eye_pad),
        (RIGHT_EYE_REGION, eye_pad),
        (LIPS_PTS, mouth_pad),
    ]:
        bbox = region_bbox(landmark_arr, region_indices, w, h, pad)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        # 1) rotated patch (full bbox)
        patch = result[y1:y2, x1:x2].copy()
        patch_rot = cv2.rotate(patch, cv2.ROTATE_180)
        # 2) contour-shaped alpha mask in bbox-local coords, then rotate
        # the mask 180° too so it matches the rotated patch's geometry.
        mask = contour_polygon_mask(landmark_arr, region_indices, bbox,
                                     dilate_px, feather_sigma)
        # Rotate the mask the same 180° so the mask still encloses the feature
        # after the patch is flipped. (cv2.rotate ROTATE_180 = vertical+horizontal flip)
        mask_rot = cv2.rotate(mask, cv2.ROTATE_180)
        # Effective mask = element-wise max of original-orientation mask and
        # rotated mask, so the fade boundary is symmetric and covers any
        # pixel that's "feature-content" in either orientation.
        eff = np.maximum(mask, mask_rot)[..., None]
        base = result[y1:y2, x1:x2].astype(np.float32)
        new = patch_rot.astype(np.float32)
        blended = base * (1.0 - eff) + new * eff
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def rotate_180(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.rotate(img_rgb, cv2.ROTATE_180)


def iter_ffhq_shards(shard_dir: Path, shuffle: bool, seed: int):
    tars = sorted(shard_dir.glob("*.tar"))
    if not tars:
        raise FileNotFoundError(f"no tar shards in {shard_dir}")
    rng = random.Random(seed)
    for tar_path in tars:
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".webp")]
            if shuffle:
                rng.shuffle(members)
            for m in members:
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                except Exception:
                    continue
                yield m.name.replace(".webp", ""), img


def iter_gpt_images(gpt_dir: Path):
    """Yield (source_name, PIL.Image) for already-generated PNGs in gpt_dir."""
    pngs = sorted(gpt_dir.glob("identity_*.png"))
    for p in pngs:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        yield p.stem, img


def main(args):
    out_root = Path(args.output_root)
    identities_dir = out_root / "identities"
    landmarks_dir = out_root / "identity_landmarks"
    thatcher_dir = out_root / "thatcher"
    for d in (identities_dir, landmarks_dir, thatcher_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.gpt_image_dir:
        source_iter = iter_gpt_images(Path(args.gpt_image_dir))
        source_label = f"gpt:{args.gpt_image_dir}"
    elif args.ffhq_shard_dir:
        source_iter = iter_ffhq_shards(Path(args.ffhq_shard_dir),
                                        shuffle=True, seed=args.seed)
        source_label = f"ffhq:{args.ffhq_shard_dir}"
    else:
        raise SystemExit("provide either --gpt_image_dir or --ffhq_shard_dir")

    print(f"[init] source={source_label}, target_n={args.target_n}")
    print(f"[init] contour dilate_px={args.dilate_px}, feather_sigma={args.feather_sigma}")

    identity_records: list[Identity] = []
    manifest_rows: list[dict] = []

    with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                         refine_landmarks=True,
                         min_detection_confidence=0.7) as mp_face:
        pbar = tqdm(source_iter, desc="thatcher_v2", total=args.scan_cap)
        n_seen = 0
        n_reject_lm = 0; n_reject_iod = 0; n_reject_tilt = 0
        for source_name, pil_img in pbar:
            n_seen += 1
            if n_seen > args.scan_cap or len(identity_records) >= args.target_n:
                break
            rgb = np.array(pil_img)
            h, w = rgb.shape[:2]
            res = mp_face.process(rgb)
            if not res.multi_face_landmarks:
                n_reject_lm += 1; continue
            lms = res.multi_face_landmarks[0].landmark
            lm_arr = landmarks_to_array(lms, w, h)
            try:
                le = lm_arr[LEFT_EYE_PTS].mean(axis=0)
                re = lm_arr[RIGHT_EYE_PTS].mean(axis=0)
            except IndexError:
                n_reject_lm += 1; continue
            iod = float(np.linalg.norm(le - re))
            if iod < args.min_iod_px:
                n_reject_iod += 1; continue
            dy = re[1] - le[1]; dx = re[0] - le[0]
            tilt = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
            if tilt > args.max_tilt_deg:
                n_reject_tilt += 1; continue

            identity_id = len(identity_records)
            eye_pad = max(8, int(min(h, w) * 0.025))
            mouth_pad = max(10, int(min(h, w) * 0.030))
            v1 = rgb.copy()
            v2 = thatcherize_contour(rgb, lm_arr, eye_pad, mouth_pad,
                                      args.dilate_px, args.feather_sigma)
            v3 = rotate_180(rgb)
            v4 = rotate_180(v2)

            canonical_path = identities_dir / f"identity_{identity_id:04d}.png"
            Image.fromarray(rgb).save(canonical_path)
            lm_json = landmarks_dir / f"identity_{identity_id:04d}.json"
            with open(lm_json, "w") as f:
                json.dump({
                    "identity_id": identity_id, "source_name": source_name,
                    "native_size": [int(w), int(h)],
                    "landmarks": [[float(x), float(y)] for x, y in lm_arr.tolist()],
                    "iod_px": iod, "tilt_deg": tilt,
                    "eye_pad_used": eye_pad, "mouth_pad_used": mouth_pad,
                    "dilate_px_used": args.dilate_px,
                    "feather_sigma_used": args.feather_sigma,
                }, f)
            identity_records.append(Identity(
                identity_id=identity_id, source_name=source_name,
                canonical_png_path=str(canonical_path),
                landmarks_json_path=str(lm_json),
            ))
            for vname, vimg, orient, thatch in [
                ("V1_upright_normal",   v1, "upright",  False),
                ("V2_upright_thatched", v2, "upright",  True),
                ("V3_inverted_normal",  v3, "inverted", False),
                ("V4_inverted_thatched",v4, "inverted", True),
            ]:
                p = thatcher_dir / f"identity_{identity_id:04d}_{vname}.png"
                Image.fromarray(vimg).save(p)
                manifest_rows.append({
                    "stim_id": f"thatcher_id{identity_id:04d}_{vname}",
                    "identity_id": identity_id,
                    "lfw_name": source_name,
                    "paradigm": "thatcher",
                    "condition": vname,
                    "orientation": orient,
                    "thatcherized": int(thatch),
                    "path": str(p),
                })
            pbar.set_postfix(accepted=len(identity_records))

    print(f"[done] {len(identity_records)} accepted (scanned ~{n_seen}; "
          f"reject lm={n_reject_lm} iod={n_reject_iod} tilt={n_reject_tilt})")
    with open(out_root / "thatcher_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    with open(out_root / "identity_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(identity_records[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in identity_records])
    print(f"  manifests: {out_root}/{{thatcher_manifest.csv, identity_manifest.csv}}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ffhq_shard_dir", default=None,
                   help="FFHQ tar shard dir (use this OR --gpt_image_dir)")
    p.add_argument("--gpt_image_dir", default=None,
                   help="dir of pre-generated GPT source identity_*.png")
    p.add_argument("--output_root", required=True)
    p.add_argument("--target_n", type=int, default=200)
    p.add_argument("--scan_cap", type=int, default=2000,
                   help="cap on inputs to scan when sourcing from FFHQ")
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--min_iod_px", type=float, default=120.0)
    p.add_argument("--max_tilt_deg", type=float, default=15.0)
    p.add_argument("--dilate_px", type=int, default=4,
                   help="contour mask dilation (px) before feather")
    p.add_argument("--feather_sigma", type=float, default=4.0,
                   help="Gaussian sigma (px) for soft contour edge")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
