"""stimuli/classical_cv/streaming_phase_a.py — Phase A that watches a
download directory and processes tars as they finalize, in parallel.

Watches data/composite_ffhq_src/ for *.tar files growing past `min_size`
(default 1.5 GB = a complete Thatcher tar). Each newly-complete tar is
handed to a worker pool that runs the full Phase A attrs extraction on
each identity inside, appending to data/composite_attrs.pkl.

Exits when all 70 tars have been processed AND the download directory
contains 70 complete files.

Usage:
    python -m stimuli.classical_cv.streaming_phase_a \
        --src data/composite_ffhq_src --attrs_pkl data/composite_attrs.pkl \
        --workers 16 --target_tars 70
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

from stimuli.classical_cv.composite_v2_parallel import (
    _phase_a_init, _phase_a_one, _iter_thatcher_tar_v1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/composite_ffhq_src")
    ap.add_argument("--attrs_pkl", default="data/composite_attrs.pkl")
    ap.add_argument("--predictor",
                    default="models/shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--recognizer",
                    default="models/dlib_face_recognition_resnet_model_v1.dat")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min_size", type=int, default=1_500_000_000)
    ap.add_argument("--target_tars", type=int, default=70)
    ap.add_argument("--no_insight", action="store_true")
    ap.add_argument("--check_interval", type=float, default=30.0)
    args = ap.parse_args()

    src = Path(args.src)
    attrs_path = Path(args.attrs_pkl)
    attrs_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing attrs if any (resume)
    if attrs_path.exists():
        with open(attrs_path, "rb") as f:
            attrs = pickle.load(f)
        print(f"[stream-A] resume: {len(attrs)} attrs already in {attrs_path}")
    else:
        attrs = []
    processed_names = {a["name"] for a in attrs if "name" in a}
    processed_tars: set[str] = set()
    # Note: we track tars by filename, but actually re-scan identities each time

    print(f"[stream-A] watching {src} for complete tars (size >= {args.min_size/1e9:.1f} GB)")
    print(f"[stream-A] workers={args.workers}  insight={'OFF' if args.no_insight else 'ON'}")

    pool = mp.Pool(processes=args.workers,
                    initializer=_phase_a_init,
                    initargs=(args.predictor, args.recognizer, not args.no_insight))

    t_start = time.monotonic()
    last_save = t_start
    while True:
        complete_tars = sorted(
            tp for tp in src.glob("*.tar") if tp.stat().st_size >= args.min_size
        )
        new_tars = [tp for tp in complete_tars if tp.name not in processed_tars]
        if new_tars:
            print(f"[stream-A] {len(new_tars)} new complete tars: "
                  f"{[t.name for t in new_tars[:3]]}{'...' if len(new_tars)>3 else ''}",
                  flush=True)
            for tp in new_tars:
                t0 = time.monotonic()
                tasks = []
                for name, png, lm in _iter_thatcher_tar_v1(tp):
                    if name not in processed_names:
                        tasks.append((name, png, lm))
                if not tasks:
                    processed_tars.add(tp.name)
                    continue
                n_kept = 0
                for r in pool.imap_unordered(_phase_a_one, tasks, chunksize=4):
                    if r is not None and "error" not in r:
                        attrs.append(r)
                        processed_names.add(r["name"])
                        n_kept += 1
                el = time.monotonic() - t0
                rate = len(tasks) / max(el, 1)
                print(f"  [{tp.name}] tasks={len(tasks)} kept={n_kept} "
                      f"in {el:.0f}s ({rate:.1f}/s)  total_kept={len(attrs)}",
                      flush=True)
                processed_tars.add(tp.name)
            # Save attrs every time we finish processing some
            with open(attrs_path, "wb") as f:
                pickle.dump(attrs, f, protocol=4)
            sz_mb = attrs_path.stat().st_size / 1e6
            print(f"  saved {attrs_path}  ({sz_mb:.1f} MB, {len(attrs)} entries)",
                  flush=True)
            last_save = time.monotonic()
        elif len(processed_tars) >= args.target_tars:
            print(f"[stream-A] DONE: processed {len(processed_tars)} tars, "
                  f"kept {len(attrs)} attrs", flush=True)
            break
        else:
            done = len(processed_tars)
            in_dir = len(list(src.glob("*.tar")))
            print(f"[stream-A] waiting … processed={done}/{args.target_tars}  "
                  f"all-tars-in-dir={in_dir}  elapsed="
                  f"{(time.monotonic()-t_start)/60:.1f}min",
                  flush=True)
            time.sleep(args.check_interval)

    pool.close(); pool.join()
    # Apply load-time filters (infant, yaw, mouth-offset)
    n_inf = sum(1 for r in attrs if r["lwr_upr_ratio"] < 3.85)
    n_yaw = sum(1 for r in attrs if abs(r["yaw_deg"]) > 15.0)
    n_mouth = sum(1 for r in attrs if r["mouth_offset_ratio"] > 0.03)
    kept = [r for r in attrs if (r["lwr_upr_ratio"] >= 3.85
                                  and abs(r["yaw_deg"]) <= 15.0
                                  and r["mouth_offset_ratio"] <= 0.03)]
    print(f"\n[stream-A] FILTER: total={len(attrs)}  "
          f"-{n_inf} infant  -{n_yaw} yaw>15°  -{n_mouth} mouth-off-axis")
    print(f"  kept after filter: {len(kept)}")
    # Overwrite with filtered version
    with open(attrs_path, "wb") as f:
        pickle.dump(kept, f, protocol=4)
    print(f"  saved filtered: {attrs_path}  ({attrs_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
