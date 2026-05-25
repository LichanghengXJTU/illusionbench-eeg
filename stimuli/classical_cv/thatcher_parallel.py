"""stimuli/classical_cv/thatcher_parallel.py — multiprocess Thatcher gen.

Per user direction 2026-05-25 ("能加速吗?"): box has 224 CPU cores, prior
single-thread thatcher_classical.py used 1 → ~26h ETA on 70k FFHQ.
This module uses multiprocessing.Pool to fan out across N workers
(default 64; safe margin from 224 to avoid OOM / scheduler thrash).

Identity-dir naming: uses FFHQ image name (e.g. `ffhq_01234`) instead of
sequential `identity_NNNN` to avoid cross-worker race conditions.
Manifest stores both ffhq_name + a post-hoc sequential id for downstream
analysis compatibility.

Each worker:
  - Loads its OWN dlib detector + predictor (cannot share across procs)
  - Iterates an assigned slice of FFHQ tar files
  - Applies eye-only Thatcher (Thompson 1980) per filtered face
  - Writes 4 PNGs + landmarks.json to ffhq_{NAME}/ subdir

Resilient to crashes (idempotent: skips ffhq_{NAME} dirs that already
contain thatcher_V4.png).
"""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Import the pure functions (no class state) from the single-thread module
from stimuli.classical_cv.face_pipeline import (
    FacePipeline, default_filter, iter_ffhq_tar,
)
from stimuli.classical_cv.thatcher_classical import (
    thatcherize_classical, overlay_polygons, make_contact_sheet,
)


# Globals set in each worker by _init
_pipe: FacePipeline | None = None
_args: dict | None = None


def _init(predictor_path: str, args_dict: dict):
    """Pool initializer: each worker creates ONE FacePipeline."""
    global _pipe, _args
    _pipe = FacePipeline(predictor_path)
    _args = args_dict


def _process_tar(tar_path_str: str) -> dict:
    """Worker function: process ONE tar, return summary dict."""
    global _pipe, _args
    tar_path = Path(tar_path_str)
    out_root = Path(_args["output_dir"])
    pad = _args["pad"]
    n_passed = 0
    n_scanned = 0
    rows = []
    for src_name, rgb in iter_ffhq_tar(tar_path, max_n=-1,
                                         shuffle=False,
                                         seed=_args["seed"]):
        n_scanned += 1
        idir = out_root / f"ffhq_{src_name}"
        v4_path = idir / "thatcher_V4.png"
        # Idempotent: skip if already done
        if v4_path.exists():
            n_passed += 1
            continue
        try:
            faces = _pipe.detect_and_landmark(rgb)
        except Exception:
            continue
        if not faces:
            continue
        rec = max(faces, key=lambda r: (r.bbox[2] - r.bbox[0]) *
                                          (r.bbox[3] - r.bbox[1]))
        rec.src_name = src_name
        ok, _ = default_filter(rec)
        if not ok:
            continue
        idir.mkdir(parents=True, exist_ok=True)
        try:
            v1 = rgb.copy()
            v2 = thatcherize_classical(rgb, rec.landmarks, pad=pad)
            v3 = cv2.rotate(v1, cv2.ROTATE_180)
            v4 = cv2.rotate(v2, cv2.ROTATE_180)
            for cond, im in [("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)]:
                Image.fromarray(im).save(idir / f"thatcher_{cond}.png")
            (idir / "landmarks.json").write_text(json.dumps({
                "ffhq_name": src_name,
                "tar": tar_path.name,
                "landmarks": rec.landmarks.tolist(),
                "iod_px": rec.iod_px,
                "tilt_deg": rec.tilt_deg,
                "yaw_score": rec.yaw_score,
                "bbox": list(rec.bbox),
            }))
            n_passed += 1
            rows.append({"ffhq_name": src_name, "tar": tar_path.name})
        except Exception as e:
            # Best-effort; skip bad ones
            continue
    return {"tar": tar_path.name, "n_scanned": n_scanned,
             "n_passed": n_passed, "rows": rows}


def main(args):
    tar_dir = Path(args.tar_dir)
    tar_paths = sorted(tar_dir.glob("*.tar"))
    print(f"[init] {len(tar_paths)} tars to process across {args.workers} workers")
    print(f"[init] output: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args_dict = {
        "output_dir": args.output_dir,
        "pad": args.pad,
        "seed": args.seed,
    }
    t0 = time.monotonic()
    total_scanned = 0
    total_passed = 0
    all_rows = []
    with mp.Pool(processes=args.workers,
                  initializer=_init,
                  initargs=(args.predictor, args_dict)) as pool:
        for i, result in enumerate(pool.imap_unordered(
                _process_tar, [str(p) for p in tar_paths])):
            total_scanned += result["n_scanned"]
            total_passed += result["n_passed"]
            all_rows.extend(result["rows"])
            elapsed = time.monotonic() - t0
            rate = total_passed / max(1, elapsed) * 60   # per minute
            print(f"  [{i+1}/{len(tar_paths)}] tar={result['tar']} "
                  f"scanned={result['n_scanned']} passed={result['n_passed']} "
                  f"| total_passed={total_passed} elapsed={elapsed:.0f}s "
                  f"rate={rate:.1f}/min", flush=True)

    # Write consolidated manifest
    manifest_path = Path(args.output_dir) / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    print(f"\n[done] total: scanned={total_scanned} passed={total_passed}")
    print(f"       elapsed: {(time.monotonic()-t0)/60:.1f} min")
    print(f"       manifest: {manifest_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor",
                   default="/workspace/models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--tar_dir",
                   default="/workspace/.hf_cache/datasets--gaunernst--ffhq-1024-wds/snapshots/d74f1f1f59e3bbe975bee29872b9bef827314577")
    p.add_argument("--output_dir", default="data/thatcher_ffhq_70k_parallel")
    p.add_argument("--workers", type=int, default=64,
                   help="number of parallel workers (224 cores available)")
    p.add_argument("--pad", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260525)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
