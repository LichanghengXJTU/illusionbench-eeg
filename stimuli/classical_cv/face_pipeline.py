"""stimuli/classical_cv/face_pipeline.py — dlib face + landmark + filter.

Pure classical CV pipeline (per user direction 2026-05-25):
  - dlib HOG+SVM face detector (Dalal 2005 + Cortes 1995, pre-DL)
  - dlib 68-point shape predictor (Kazemi 2014, ensemble regression trees)
  - Filter: face dominant, frontal, neutral-or-mild expression, no glasses, IPD ok

Used by:
  - thatcher_classical.py    (eye/mouth polygon segmentation)
  - composite_classical.py   (nose-tip cut + IPD-rigid alignment)
  - partwhole_classical.py   (eye region polygon segmentation)
  - demographic_cluster.py   (HSV + geometric descriptors)

All landmark indices follow the 300-W 68-point convention:
  0-16   = jawline
  17-21  = left eyebrow
  22-26  = right eyebrow
  27-30  = nose bridge
  30     = nose tip
  31-35  = nose nostrils
  36-41  = left eye
  42-47  = right eye
  48-67  = mouth (outer 48-59, inner 60-67)

(Note: dlib uses subject-frame left/right — i.e., the SUBJECT's left eye is on
the VIEWER's right side. We follow this convention throughout.)
"""
from __future__ import annotations
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import dlib
import numpy as np
from PIL import Image


# === Landmark index groups (68-point Multi-PIE/iBUG-300W convention) ===
JAW         = list(range(0, 17))
LEFT_BROW   = list(range(17, 22))
RIGHT_BROW  = list(range(22, 27))
NOSE_BRIDGE = list(range(27, 31))
NOSE_TIP    = 30
NOSE_BASE   = list(range(31, 36))
LEFT_EYE    = list(range(36, 42))
RIGHT_EYE   = list(range(42, 48))
MOUTH_OUTER = list(range(48, 60))
MOUTH_INNER = list(range(60, 68))
LEFT_EYE_REGION  = LEFT_BROW  + LEFT_EYE
RIGHT_EYE_REGION = RIGHT_BROW + RIGHT_EYE
BOTH_EYES_REGION = LEFT_EYE_REGION + RIGHT_EYE_REGION
MOUTH_REGION     = MOUTH_OUTER + MOUTH_INNER


@dataclass
class FaceRecord:
    image: np.ndarray            # RGB uint8 (H, W, 3)
    landmarks: np.ndarray        # (68, 2) float32 pixel coords
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    iod_px: float                # inter-ocular distance
    tilt_deg: float              # absolute eye-line tilt from horizontal
    yaw_score: float             # eye-symmetry yaw proxy (0 = frontal)
    face_area_frac: float        # bbox area / total image area
    glasses_edge_density: float  # Canny density inside eye bbox (proxy)
    mouth_open_frac: float       # inner-lip h / mouth-width (smile proxy)
    src_name: str                # FFHQ filename root


class FacePipeline:
    """Wrap dlib detector + 68-pt predictor + classical-CV filters.

    Single instance — heavy models loaded once."""

    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_and_landmark(self, rgb: np.ndarray) -> list[FaceRecord]:
        """Return all detected faces with landmarks (usually 1 face per FFHQ)."""
        h, w = rgb.shape[:2]
        dets = self.detector(rgb, 1)   # upsample once for small faces
        records: list[FaceRecord] = []
        for det in dets:
            shape = self.predictor(rgb, det)
            lm = np.array([[shape.part(i).x, shape.part(i).y]
                           for i in range(68)], dtype=np.float32)
            x1 = int(max(0, det.left()))
            y1 = int(max(0, det.top()))
            x2 = int(min(w, det.right()))
            y2 = int(min(h, det.bottom()))
            rec = self._metrics(rgb, lm, (x1, y1, x2, y2), "")
            records.append(rec)
        return records

    def _metrics(self, rgb: np.ndarray, lm: np.ndarray,
                  bbox: tuple[int, int, int, int],
                  src_name: str) -> FaceRecord:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        le_c = lm[LEFT_EYE].mean(axis=0)
        re_c = lm[RIGHT_EYE].mean(axis=0)
        iod = float(np.linalg.norm(le_c - re_c))
        dy = re_c[1] - le_c[1]; dx = re_c[0] - le_c[0]
        tilt = float(np.degrees(np.arctan2(abs(dy), abs(dx))))

        # Yaw proxy: ratio (eye→nose-tip-along-eye-line) on the left vs right
        # half. For frontal face, ratios are equal → yaw_score = 0.
        nose = lm[NOSE_TIP]
        eye_mid = (le_c + re_c) / 2
        # signed offset of nose along the perpendicular to eye line
        eye_axis = (re_c - le_c)
        eye_axis_n = eye_axis / max(1e-6, np.linalg.norm(eye_axis))
        perp = np.array([-eye_axis_n[1], eye_axis_n[0]])
        nose_offset = float((nose - eye_mid) @ perp)
        # Normalize by IOD for scale invariance. Frontal face: nose_offset/iod ≈ 0
        # is wrong heuristic. Better: distance from nose to the eye-midline,
        # NORMALIZED. The thing that matters for yaw is left/right eye-to-nose
        # ratio asymmetry.
        d_lnose = float(np.linalg.norm(le_c - nose))
        d_rnose = float(np.linalg.norm(re_c - nose))
        yaw_score = abs(d_lnose - d_rnose) / max(d_lnose, d_rnose, 1e-6)

        face_area_frac = ((x2 - x1) * (y2 - y1)) / (h * w)

        # Glasses proxy: Canny edge density in eye-bbox region.
        ex1 = int(max(0, min(le_c[0], re_c[0]) - iod * 0.3))
        ex2 = int(min(w, max(le_c[0], re_c[0]) + iod * 0.3))
        ey1 = int(max(0, min(le_c[1], re_c[1]) - iod * 0.4))
        ey2 = int(min(h, max(le_c[1], re_c[1]) + iod * 0.4))
        glasses_density = 0.0
        if ex2 > ex1 + 2 and ey2 > ey1 + 2:
            gray = cv2.cvtColor(rgb[ey1:ey2, ex1:ex2], cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 180)
            glasses_density = float(edges.mean()) / 255.0

        # Mouth open / smile proxy: ratio inner-lip-vertical-gap / mouth-width
        mouth = lm[MOUTH_REGION]
        mw = float(mouth[:, 0].max() - mouth[:, 0].min())
        inner_top = lm[MOUTH_INNER[:5]][:, 1].mean()   # 60-64 = upper inner lip
        inner_bot = lm[MOUTH_INNER[5:]][:, 1].mean()   # 65-67 = lower inner lip
        inner_h = float(abs(inner_bot - inner_top))
        mouth_open = inner_h / max(mw, 1e-6)

        return FaceRecord(
            image=rgb, landmarks=lm, bbox=bbox, iod_px=iod, tilt_deg=tilt,
            yaw_score=yaw_score, face_area_frac=face_area_frac,
            glasses_edge_density=glasses_density,
            mouth_open_frac=mouth_open, src_name=src_name,
        )


def default_filter(rec: FaceRecord) -> tuple[bool, dict]:
    """Apply standard filter. Returns (pass, per-criterion-bool-dict)."""
    criteria = {
        "iod_ok":          rec.iod_px >= 120.0,
        "tilt_ok":         rec.tilt_deg <= 15.0,
        "yaw_ok":          rec.yaw_score <= 0.18,   # ~< 15° yaw
        "face_dominant":   rec.face_area_frac >= 0.20,
        "no_glasses":      rec.glasses_edge_density <= 0.15,
        "mouth_not_wide_open": rec.mouth_open_frac <= 0.18,
        # Note: we ALLOW smiling/closed-mouth smile (mouth_open_frac small);
        # we only reject loudly-laughing/yawning (large inner-lip gap).
    }
    return all(criteria.values()), criteria


def iter_ffhq_tar(tar_path: Path, max_n: int = -1, shuffle: bool = False,
                   seed: int = 20260525) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (basename, rgb_uint8) from one FFHQ webdataset tar."""
    import random
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".webp")]
        if shuffle:
            random.Random(seed).shuffle(members)
        n = 0
        for m in members:
            if 0 < max_n <= n:
                break
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                pil = Image.open(io.BytesIO(f.read())).convert("RGB")
            except Exception:
                continue
            yield m.name.replace(".webp", ""), np.array(pil)
            n += 1


def sanity_run(predictor_path: str, ffhq_tar: Path,
                n_scan: int = 50, out_dir: Path | None = None) -> dict:
    """Phase 1 sanity: scan N FFHQ images, report filter pass rate."""
    pipe = FacePipeline(predictor_path)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    n_scanned = 0; n_face_detected = 0; n_passed = 0
    per_crit_fail = {k: 0 for k in [
        "iod_ok", "tilt_ok", "yaw_ok", "face_dominant",
        "no_glasses", "mouth_not_wide_open"]}
    rows = []
    for src_name, rgb in iter_ffhq_tar(ffhq_tar, max_n=n_scan, shuffle=True):
        n_scanned += 1
        faces = pipe.detect_and_landmark(rgb)
        if not faces:
            rows.append({"name": src_name, "n_faces": 0, "passed": False})
            continue
        n_face_detected += 1
        rec = max(faces, key=lambda r: (r.bbox[2] - r.bbox[0]) *
                                         (r.bbox[3] - r.bbox[1]))
        rec.src_name = src_name
        passed, crit = default_filter(rec)
        if passed:
            n_passed += 1
        for k, ok in crit.items():
            if not ok:
                per_crit_fail[k] += 1
        rows.append({
            "name": src_name, "n_faces": len(faces),
            "iod": rec.iod_px, "tilt": rec.tilt_deg,
            "yaw": rec.yaw_score, "face_area_frac": rec.face_area_frac,
            "glasses_density": rec.glasses_edge_density,
            "mouth_open": rec.mouth_open_frac,
            "passed": passed,
        })
        if out_dir is not None and passed and n_passed <= 10:
            # save the passing image with landmark overlay
            vis = rgb.copy()
            for x, y in rec.landmarks.astype(int):
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
            x1, y1, x2, y2 = rec.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            Image.fromarray(vis).save(out_dir / f"pass_{n_passed:02d}_{src_name}.png")
        if out_dir is not None and not passed and (n_scanned - n_passed) <= 5:
            # save a few failure exemplars
            Image.fromarray(rgb).save(
                out_dir / f"fail_{n_scanned:02d}_{src_name}.png"
            )

    pass_rate = n_passed / max(1, n_scanned)
    summary = {
        "n_scanned": n_scanned,
        "n_face_detected": n_face_detected,
        "n_passed": n_passed,
        "pass_rate": pass_rate,
        "per_criterion_fail_count": per_crit_fail,
    }
    print("\n=== Phase 1 sanity summary ===")
    print(f"  scanned: {n_scanned}")
    print(f"  faces detected: {n_face_detected} ({n_face_detected/max(1,n_scanned)*100:.1f}%)")
    print(f"  passed all filters: {n_passed} ({pass_rate*100:.1f}%)")
    print(f"  per-criterion fail counts:")
    for k, v in per_crit_fail.items():
        print(f"    {k:24s} {v:3d}")
    if out_dir is not None:
        import json, csv
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        with open(out_dir / "per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  saved: {out_dir}/summary.json + per_image.csv + 15 sample PNGs")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--predictor",
                   default="/workspace/models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--tar",
                   default="/workspace/.hf_cache/hub/datasets--gaunernst--ffhq-1024-wds/snapshots/d74f1f1f59e3bbe975bee29872b9bef827314577/00000.tar")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", default="data/classical_cv_phase1_sanity")
    args = p.parse_args()
    sanity_run(args.predictor, Path(args.tar), n_scan=args.n,
                out_dir=Path(args.out))
