"""stimuli/classical_cv/composite_v2_parallel.py — full-scale Young 1987
composite generator. Phase A (per-identity attrs) + Phase B (per-template
composite generation), both multiprocessed.

Reuses the pilot pipeline (composite_v2_pilot.py) for the actual composite
construction, but adds:
  - Streaming source from Thatcher tars on HF
  - Parallel Phase A with cached attrs.pkl
  - Parallel Phase B writing directly to webdataset tars
  - Output: v2.0/composite_NNN.tar files, one shard per ~500 (T, D) pairs

Phase A loads ALL identities from data/composite_ffhq_src/*.tar, runs:
  dlib 68 landmarks (already in landmarks.json from Thatcher pipeline)
  dlib 128-d embedding
  MediaPipe Face Mesh 478 landmarks + facial_transformation_matrix
  InsightFace gender / age
  Skin LAB, face_w, cheek_w_at_cut, mouth_offset_ratio,
  cut_to_chin_ratio, lwr_upr_ratio, yaw_deg

Phase B loads attrs.pkl, picks templates passing all filters, for each
template picks K=3 donors via the same filter stack as composite_v2_pilot,
then runs make_composite to generate 4 PNGs per (T, D). Packs into
webdataset tars.

Usage:
    cd ~/Desktop/EEG/illusionbench
    source .venv/bin/activate
    python -m stimuli.classical_cv.composite_v2_parallel \\
        --phase=both --workers=20 --donors_per_template=3 \\
        --tar_src=data/composite_ffhq_src --tar_dst=data/composite_v2 \\
        --attrs_pkl=data/composite_attrs.pkl
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import multiprocessing as mp
import pickle
import random
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stimuli.classical_cv.face_embedding import FaceEmbedder
from stimuli.classical_cv.composite_v2_pilot import (
    IdAttr, compute_attrs, pick_donors, make_composite, save_result,
    detect_mp_landmarks, decompose_head_pose, cheek_x_at_y,
    LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, JAW, NOSE_BOTTOM_IDX,
    UPPER_LIP_CENTER_IDX,
)


# === Phase A: per-identity attribute extraction =====================

_embedder = None
_insight = None


def _phase_a_init(predictor_path: str, recognizer_path: str,
                  use_insight: bool):
    """Per-worker initializer. Loads dlib + InsightFace once per process."""
    global _embedder, _insight
    _embedder = FaceEmbedder(predictor_path, recognizer_path)
    if use_insight:
        from stimuli.classical_cv.insight_attrs import InsightAttrExtractor
        _insight = InsightAttrExtractor()


def _phase_a_one(task: tuple[str, bytes, bytes]) -> dict | None:
    """Worker function: takes (name, png_bytes, landmarks_json_str) and
    returns a serializable attrs dict (NO rgb / lm arrays — we re-load
    from disk in Phase B)."""
    global _embedder, _insight
    name, png_bytes, lm_json = task
    try:
        rgb = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        meta = json.loads(lm_json)
        lm = np.array(meta["landmarks"], dtype=np.float32)
        attr = compute_attrs(name, rgb, lm, _embedder, insight=_insight)
        if attr is None:
            return None
        return {
            "name": attr.name,
            "lm": attr.lm.tolist(),
            "mp_lm": attr.mp_lm.tolist(),
            "emb": attr.emb.tolist(),
            "skin_lab": attr.skin_lab.tolist(),
            "face_w": attr.face_w,
            "face_h": attr.face_h,
            "bottom_w": attr.bottom_w,
            "bottom_face_w_ratio": attr.bottom_face_w_ratio,
            "y_cut": attr.y_cut,
            "lwr_upr_ratio": attr.lwr_upr_ratio,
            "yaw_deg": attr.yaw_deg,
            "pitch_deg": attr.pitch_deg,
            "roll_deg": attr.roll_deg,
            "mouth_offset_ratio": attr.mouth_offset_ratio,
            "cheek_w_at_cut": attr.cheek_w_at_cut,
            "cut_to_chin_ratio": attr.cut_to_chin_ratio,
            "predicted_age": attr.predicted_age,
            "predicted_gender": attr.predicted_gender,
        }
    except Exception as e:
        return {"name": name, "error": repr(e)}


def _iter_thatcher_tar_v1(tar_path: Path):
    """Yield (ffhq_name, png_bytes, landmarks_json_str) tuples for every
    identity in one Thatcher tar shard.

    Robust to PARTIAL tars (mid-download): iterates members one at a time
    with `tf.next()` instead of `tf.getmembers()` (which loads the entire
    member index upfront and raises ReadError on EOF). Streams V1+landmarks
    bytes for each complete identity, breaks cleanly when the tar ends
    (whether by EOF block or by unexpected end-of-data).

    Thatcher tar layout:
        ffhq_NNNNN/V1_upright_normal.png   (= original FFHQ face)
        ffhq_NNNNN/V2_upright_thatched.png
        ffhq_NNNNN/V3_inverted_normal.png
        ffhq_NNNNN/V4_inverted_thatched.png
        ffhq_NNNNN/landmarks.json
    We only need V1 + landmarks.
    """
    pending: dict[str, dict[str, bytes]] = {}
    try:
        tf = tarfile.open(tar_path, "r")
    except Exception:
        return
    try:
        while True:
            try:
                m = tf.next()
            except (tarfile.ReadError, EOFError, OSError):
                break
            if m is None:
                break
            if "/" not in m.name:
                continue
            ident, fname = m.name.split("/", 1)
            if fname not in ("V1_upright_normal.png", "landmarks.json"):
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                content = f.read()
            except (tarfile.ReadError, EOFError, OSError):
                break
            pending.setdefault(ident, {})[fname] = content
            sub = pending[ident]
            if "V1_upright_normal.png" in sub and "landmarks.json" in sub:
                name = ident.replace("ffhq_", "")
                yield (name,
                       sub["V1_upright_normal.png"],
                       sub["landmarks.json"].decode("utf-8"))
                del pending[ident]
    finally:
        try:
            tf.close()
        except Exception:
            pass


def run_phase_a(args):
    src = Path(args.tar_src)
    tars = sorted(src.glob("*.tar"))
    print(f"[phase A] {len(tars)} Thatcher tars under {src}")
    if not tars:
        raise SystemExit(f"No tars in {src}")
    # Gather work items: ALL identities across ALL tars
    print(f"[phase A] indexing identities from tars ...")
    tasks = []
    t0 = time.monotonic()
    for tp in tars:
        for name, png_bytes, lm_json in _iter_thatcher_tar_v1(tp):
            tasks.append((name, png_bytes, lm_json))
    print(f"[phase A] {len(tasks)} identities indexed in {(time.monotonic()-t0):.1f}s")

    print(f"[phase A] running attrs extraction with {args.workers} workers ...")
    results: list[dict] = []
    t1 = time.monotonic()
    with mp.Pool(processes=args.workers,
                  initializer=_phase_a_init,
                  initargs=(args.predictor, args.recognizer, not args.no_insight)) as pool:
        for i, r in enumerate(pool.imap_unordered(_phase_a_one, tasks, chunksize=4)):
            if r is None:
                continue
            if "error" in r:
                continue
            results.append(r)
            if (i + 1) % 500 == 0:
                el = time.monotonic() - t1
                rate = (i + 1) / max(el, 1)
                eta = (len(tasks) - i - 1) / max(rate, 1) / 60
                print(f"  [{i+1}/{len(tasks)}] kept={len(results)} "
                      f"rate={rate:.1f}/s  eta={eta:.1f}min", flush=True)

    # Apply load-time filters (infant, yaw, mouth-offset) here
    n_inf = sum(1 for r in results if r["lwr_upr_ratio"] < 3.85)
    n_yaw = sum(1 for r in results if abs(r["yaw_deg"]) > 15.0)
    n_mouth = sum(1 for r in results if r["mouth_offset_ratio"] > 0.03)
    kept = [r for r in results if (r["lwr_upr_ratio"] >= 3.85
                                    and abs(r["yaw_deg"]) <= 15.0
                                    and r["mouth_offset_ratio"] <= 0.03)]
    print(f"\n[phase A] DONE: extracted {len(results)} / {len(tasks)}")
    print(f"  filter: -{n_inf} infants  -{n_yaw} yaw>15°  -{n_mouth} mouth-off-axis")
    print(f"  kept after filter: {len(kept)}")
    print(f"  elapsed: {(time.monotonic()-t1)/60:.1f} min")

    attrs_path = Path(args.attrs_pkl)
    attrs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(attrs_path, "wb") as f:
        pickle.dump(kept, f, protocol=4)
    print(f"  saved: {attrs_path}  ({attrs_path.stat().st_size/1e6:.1f} MB)")


# === Phase B: per-template composite generation =====================

_attrs_dicts = None
_name_to_attr = None
_args_b = None
_tar_src = None
_attrs_idattr_pool = None   # list[IdAttr] (rgb=None) reused across workers


def _dict_to_idattr(d: dict, rgb: np.ndarray) -> IdAttr:
    return IdAttr(
        name=d["name"], rgb=rgb,
        lm=np.array(d["lm"], dtype=np.float32),
        mp_lm=np.array(d["mp_lm"], dtype=np.float32),
        emb=np.array(d["emb"], dtype=np.float32),
        skin_lab=np.array(d["skin_lab"], dtype=np.float32),
        face_w=d["face_w"], face_h=d["face_h"],
        bottom_w=d["bottom_w"],
        bottom_face_w_ratio=d["bottom_face_w_ratio"],
        y_cut=d["y_cut"],
        lwr_upr_ratio=d["lwr_upr_ratio"],
        yaw_deg=d["yaw_deg"], pitch_deg=d["pitch_deg"], roll_deg=d["roll_deg"],
        mouth_offset_ratio=d["mouth_offset_ratio"],
        cheek_w_at_cut=d["cheek_w_at_cut"],
        cut_to_chin_ratio=d["cut_to_chin_ratio"],
        predicted_age=d["predicted_age"],
        predicted_gender=d["predicted_gender"],
    )


# Map ffhq_name → (tar_path, member_name) for lazy PNG load in Phase B
_pngs_index: dict[str, tuple[str, str]] = {}


def _build_png_index(tar_dir: Path) -> dict[str, tuple[str, int, int]]:
    """Build index ffhq_name → (tar_path, data_offset, size). Robust to
    partial tars: iterates with tf.next(), verifies each V1 PNG's data is
    readable, and stores the byte offset of the member's DATA so _load_rgb
    can seek-read directly (avoids re-iterating tar, which fails on partial
    tars in random-access mode).
    """
    idx = {}
    for tp in sorted(tar_dir.glob("*.tar")):
        try:
            tf = tarfile.open(tp, "r")
        except Exception:
            continue
        try:
            while True:
                try:
                    m = tf.next()
                except (tarfile.ReadError, EOFError, OSError):
                    break
                if m is None:
                    break
                if not m.name.endswith("/V1_upright_normal.png"):
                    continue
                # Verify data is fully readable
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                    if len(data) != m.size or len(data) == 0:
                        continue
                except (tarfile.ReadError, EOFError, OSError):
                    break
                ident = m.name.split("/")[0].replace("ffhq_", "")
                # m.offset_data is the start of file content (after header)
                idx[ident] = (str(tp), int(m.offset_data), int(m.size))
        finally:
            try:
                tf.close()
            except Exception:
                pass
    return idx


def _phase_b_init(attrs_pkl: str, tar_src: str, png_index: dict,
                  args_b: dict):
    global _attrs_dicts, _name_to_attr, _args_b, _tar_src, _pngs_index
    global _attrs_idattr_pool
    _args_b = args_b
    _tar_src = tar_src
    _pngs_index = png_index
    with open(attrs_pkl, "rb") as f:
        _attrs_dicts = pickle.load(f)
    _name_to_attr = {d["name"]: d for d in _attrs_dicts}
    # Pre-build the rgb=None IdAttr pool ONCE per worker (used by pick_donors,
    # which only reads scalar/array fields, never rgb)
    _attrs_idattr_pool = [_dict_to_idattr(d, rgb=None) for d in _attrs_dicts]


def _load_rgb(name: str) -> np.ndarray | None:
    """Lazy-load V1 PNG by seek-read at the cached byte offset (avoids the
    full-tar iteration that fails on partial tars when called via
    `tarfile.extractfile(name)` random access)."""
    if name not in _pngs_index:
        return None
    tar_path, offset, size = _pngs_index[name]
    try:
        with open(tar_path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        if len(data) != size:
            return None
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        return None


def _process_one_template(template_idx: int) -> dict:
    """Worker function: given a template index into _attrs_dicts, generate
    all (T, donors) composites and write 4 PNG + 1 JSON per pair to the
    output directory (later packed into tars by package_composite_v2.py)."""
    global _attrs_dicts, _name_to_attr, _args_b, _attrs_idattr_pool
    try:
        t_dict = _attrs_dicts[template_idx]
        template_noimg = _attrs_idattr_pool[template_idx]
        donors_noimg = pick_donors(template_noimg, _attrs_idattr_pool,
                                    k=_args_b["donors_per_template"])
        if len(donors_noimg) < _args_b["donors_per_template"]:
            return {"template": t_dict["name"], "status": "pool_too_small",
                    "pool_size": len(donors_noimg)}
        # Lazy-load PNGs only for chosen template + donors. Tolerate
        # partial-tar failures by trying additional donors from the pool.
        template_rgb = _load_rgb(t_dict["name"])
        if template_rgb is None:
            return {"template": t_dict["name"], "status": "tar_partial_template"}
        template = _dict_to_idattr(t_dict, template_rgb)
        out_dir = Path(_args_b["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        # Idempotency: skip if all 4 PNGs of last expected donor already exist
        last_donor_name = donors_noimg[-1].name
        sentinel = out_dir / f"{t_dict['name']}__d{last_donor_name}.v4.png"
        if sentinel.exists():
            return {"template": t_dict["name"], "status": "skip_exists"}
        # Get larger pool so we can swap in extras for donors whose PNG isn't loadable
        all_candidates = pick_donors(template_noimg, _attrs_idattr_pool,
                                      k=max(20, _args_b["donors_per_template"] * 5))
        used_names: set[str] = set()
        chosen = []
        for cand in all_candidates:
            if len(chosen) >= _args_b["donors_per_template"]:
                break
            if cand.name in used_names:
                continue
            d_rgb = _load_rgb(cand.name)
            if d_rgb is None:
                continue
            try:
                donor = _dict_to_idattr(_name_to_attr[cand.name], d_rgb)
                r = make_composite(template, donor)
            except Exception as e:
                continue
            save_result(r, out_dir)
            used_names.add(cand.name)
            chosen.append(cand.name)
            n_written += 1
        if n_written < _args_b["donors_per_template"]:
            return {"template": t_dict["name"], "status": "donors_unloadable",
                    "n_donors_ok": n_written}
        return {"template": t_dict["name"], "status": "ok",
                "n_pngs": n_written * 4, "donors": chosen}
    except Exception as e:
        return {"template": _attrs_dicts[template_idx]["name"]
                if 0 <= template_idx < len(_attrs_dicts) else "unknown",
                "status": "exception", "error": repr(e)}


def run_phase_b(args):
    print(f"[phase B] loading attrs from {args.attrs_pkl}")
    with open(args.attrs_pkl, "rb") as f:
        attrs = pickle.load(f)
    print(f"[phase B] loaded {len(attrs)} kept-after-filter attrs")

    print(f"[phase B] indexing source tars under {args.tar_src}")
    png_index = _build_png_index(Path(args.tar_src))
    print(f"[phase B] indexed {len(png_index)} V1 PNGs")

    if args.max_templates:
        # Random shuffle then cap
        rng = random.Random(args.seed)
        attrs_shuf = attrs[:]
        rng.shuffle(attrs_shuf)
        attrs = attrs_shuf[:args.max_templates]
        print(f"[phase B] subsampled to {len(attrs)} templates (--max_templates)")

    args_b = {
        "out_dir": args.png_dir,
        "donors_per_template": args.donors_per_template,
        "seed": args.seed,
    }
    Path(args.png_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    counts = {"ok": 0, "pool_too_small": 0, "exception": 0,
              "exception_make": 0}
    total_pngs = 0
    with mp.Pool(processes=args.workers,
                  initializer=_phase_b_init,
                  initargs=(args.attrs_pkl, args.tar_src, png_index,
                            args_b)) as pool:
        # We re-dump attrs.pkl with full list; workers index into it via integer.
        # Pass list of template_idx that point into the attrs list.
        template_indices = list(range(len(attrs)))
        if args.max_templates:
            # already capped above
            pass
        # But the worker accesses _attrs_dicts via the SAME list. To keep
        # things consistent we re-dump the (possibly subsampled) attrs to
        # a temporary file.
        if args.max_templates:
            tmp_attrs = Path(args.attrs_pkl).with_suffix(".phaseB.pkl")
            with open(tmp_attrs, "wb") as f:
                pickle.dump(attrs, f, protocol=4)
            print(f"[phase B] wrote subsampled attrs to {tmp_attrs}")
            # Re-init pool to use subsampled pkl
            pool.close(); pool.join()
            args_b_attrs = str(tmp_attrs)
            pool = mp.Pool(processes=args.workers,
                            initializer=_phase_b_init,
                            initargs=(args_b_attrs, args.tar_src, png_index,
                                      args_b))
            # template indices now into subsampled list
            template_indices = list(range(len(attrs)))
        for i, r in enumerate(pool.imap_unordered(_process_one_template,
                                                   template_indices,
                                                   chunksize=2)):
            s = r.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
            total_pngs += r.get("n_pngs", 0)
            if (i + 1) % 50 == 0:
                el = time.monotonic() - t0
                rate = (i + 1) / max(el, 1)
                eta = (len(template_indices) - i - 1) / max(rate, 1) / 60
                print(f"  [{i+1}/{len(template_indices)}] ok={counts['ok']} "
                      f"pngs={total_pngs}  rate={rate:.2f}T/s  "
                      f"eta={eta:.1f}min", flush=True)
    print(f"\n[phase B] DONE: {counts}")
    print(f"  total PNGs: {total_pngs}")
    print(f"  elapsed: {(time.monotonic()-t0)/60:.1f} min")


# === CLI ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["a", "b", "both"], default="both")
    p.add_argument("--predictor",
                   default="models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--recognizer",
                   default="models/dlib_face_recognition_resnet_model_v1.dat")
    p.add_argument("--tar_src", default="data/composite_ffhq_src",
                   help="Directory containing Thatcher tars (with V1 + landmarks)")
    p.add_argument("--attrs_pkl", default="data/composite_attrs.pkl")
    p.add_argument("--png_dir", default="data/composite_v2_pngs",
                   help="Flat directory to dump V1-V4 PNGs + JSON per (T, D)")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--donors_per_template", type=int, default=3)
    p.add_argument("--max_templates", type=int, default=0,
                   help="If >0, subsample to this many templates")
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--no_insight", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.phase in ("a", "both"):
        run_phase_a(args)
    if args.phase in ("b", "both"):
        run_phase_b(args)


if __name__ == "__main__":
    main()
