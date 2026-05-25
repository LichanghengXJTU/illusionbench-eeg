"""stimuli/classical_cv/partwhole_parallel.py — multiprocess Part-Whole gen.

Two phases (one entry point):
  Phase A (attribute extraction): per-identity → dlib face embedding +
          skin LAB + geometry. Parallel, write to attrs.pkl.
  Phase B (swap generation): per-identity → donor match (from attrs.pkl
          in shared memory) → 3-feature swap → save 12 PNGs.

Source: data/thatcher_ffhq_70k_parallel/ffhq_NNNNN/  (26k identities;
each has thatcher_V1.png + landmarks.json from the Thatcher pipeline).

Output: data/partwhole_ffhq_70k_parallel/ffhq_NNNNN/
          partwhole_{eye,nose,mouth}_{V1,V2,V3,V4}.png  (12 files/id)
          donors.json  (records which j was used per feature, with SSIM)

Idempotent: skips identity dirs that already contain
              `partwhole_mouth_V4.png` (the LAST file written).

Usage:
  python -m stimuli.classical_cv.partwhole_parallel --phase=attrs   # phase A
  python -m stimuli.classical_cv.partwhole_parallel --phase=gen     # phase B
  python -m stimuli.classical_cv.partwhole_parallel                  # both
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from stimuli.classical_cv.face_embedding import FaceEmbedder
from stimuli.classical_cv.donor_match import build_attr, IdAttr, is_likely_infant
from stimuli.classical_cv.partwhole_classical import (
    partwhole_one_identity,
)


# Per-worker globals
_embedder: FaceEmbedder | None = None
_args: dict | None = None
_all_attrs: list[IdAttr] | None = None


# === Phase A: attribute extraction ===========================================

def _attrs_init(predictor_path: str, recognizer_path: str, args_dict: dict):
    global _embedder, _args
    _embedder = FaceEmbedder(predictor_path, recognizer_path)
    _args = args_dict


def _extract_attrs_for_id(id_dir_str: str) -> dict | None:
    """Worker function — compute attributes for one identity. Returns
    a serializable dict (NOT an IdAttr — we strip the rgb to keep pkl small)."""
    global _embedder
    id_dir = Path(id_dir_str)
    lm_path = id_dir / "landmarks.json"
    img_path = id_dir / "thatcher_V1.png"
    if not (lm_path.exists() and img_path.exists()):
        return None
    try:
        meta = json.loads(lm_path.read_text())
        rgb = np.array(Image.open(img_path).convert("RGB"))
        lm = np.array(meta["landmarks"], dtype=np.float32)
        attr = build_attr(meta["ffhq_name"], rgb, lm, _embedder)
        if attr is None:
            return None
        # Serializable form: drop rgb (re-loaded on demand in phase B)
        return {
            "name": attr.name,
            "emb": attr.emb.tolist(),
            "skin_lab": attr.skin_lab.tolist(),
            "face_w": attr.face_w, "face_h": attr.face_h,
            "face_ratio": attr.face_ratio, "nose_rel": attr.nose_rel,
            "iod_face_ratio": attr.iod_face_ratio,
            "cheek_smoothness": attr.cheek_smoothness,
            "lwr_upr_ratio": attr.lwr_upr_ratio,
            "is_infant": is_likely_infant(attr),
            "id_dir": str(id_dir),
        }
    except Exception:
        return None


def run_phase_attrs(args):
    src_root = Path(args.source_dir)
    id_dirs = sorted(src_root.glob("ffhq_*"))
    print(f"[phase A] {len(id_dirs)} identity dirs in {src_root}")
    args_dict = {}
    t0 = time.monotonic()
    results: list[dict] = []
    with mp.Pool(processes=args.workers,
                  initializer=_attrs_init,
                  initargs=(args.predictor, args.recognizer, args_dict)) as pool:
        for i, r in enumerate(pool.imap_unordered(
                _extract_attrs_for_id, [str(p) for p in id_dirs])):
            if r is not None:
                results.append(r)
            if (i + 1) % 500 == 0:
                el = time.monotonic() - t0
                print(f"  [{i+1}/{len(id_dirs)}] kept={len(results)} "
                      f"elapsed={el:.0f}s rate={(i+1)/max(1,el):.1f}/s",
                      flush=True)
    # Filter infants — these are excluded from BOTH target and donor pool
    # so they never appear in the released v1.2 dataset.
    pre_filter = len(results)
    results = [r for r in results if not r.get("is_infant", False)]
    n_infants = pre_filter - len(results)
    attrs_path = Path(args.attrs_pkl)
    attrs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(attrs_path, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"[phase A] DONE: kept {len(results)}/{len(id_dirs)} identities "
          f"(filtered {n_infants} infants/toddlers)")
    print(f"  saved: {attrs_path}  ({attrs_path.stat().st_size/1e6:.1f} MB)")
    print(f"  elapsed: {(time.monotonic()-t0)/60:.1f} min")


# === Phase B: swap generation ================================================

def _attrs_dict_to_idattr(d: dict, rgb: np.ndarray, lm: np.ndarray) -> IdAttr:
    return IdAttr(
        name=d["name"], rgb=rgb, lm=lm,
        emb=np.array(d["emb"], dtype=np.float32),
        skin_lab=np.array(d["skin_lab"], dtype=np.float32),
        face_w=d["face_w"], face_h=d["face_h"],
        face_ratio=d["face_ratio"], nose_rel=d["nose_rel"],
        iod_face_ratio=d["iod_face_ratio"],
        cheek_smoothness=d["cheek_smoothness"],
        lwr_upr_ratio=d["lwr_upr_ratio"],
    )


def _gen_init(attrs_pkl: str, args_dict: dict):
    """Phase B worker init: load attrs.pkl ONCE per worker (small ~20MB);
    keep RGB lazy-loaded on demand. attrs.pkl is already infant-filtered
    in Phase A so workers can index directly."""
    global _all_attrs, _args
    _args = args_dict
    with open(attrs_pkl, "rb") as f:
        attrs_raw = pickle.load(f)
    _all_attrs = [(d, None) for d in attrs_raw]


def _load_attr_with_rgb(attr_idx: int) -> IdAttr:
    """Lazy-load rgb + lm for an attr by index in _all_attrs."""
    d, cached = _all_attrs[attr_idx]
    if cached is not None:
        return cached
    id_dir = Path(d["id_dir"])
    rgb = np.array(Image.open(id_dir / "thatcher_V1.png").convert("RGB"))
    lm = np.array(json.loads((id_dir / "landmarks.json").read_text())["landmarks"],
                  dtype=np.float32)
    a = _attrs_dict_to_idattr(d, rgb, lm)
    _all_attrs[attr_idx] = (d, a)   # cache for this worker
    return a


def _process_one_identity(attr_idx: int) -> dict:
    """Phase B worker: generate partwhole stimuli for one identity."""
    global _args, _all_attrs
    out_root = Path(_args["output_dir"])
    seed = _args["seed"]
    i_attr = _load_attr_with_rgb(attr_idx)
    idir = out_root / f"ffhq_{i_attr.name}"
    last_file = idir / "partwhole_mouth_V4.png"
    if last_file.exists():
        return {"name": i_attr.name, "status": "skip_exists"}
    # Build a tmp list of full IdAttrs for matching. We lazy-load donors
    # only AFTER pool filtering (avoids touching disk for non-candidates).
    # For matching we need emb + skin_lab + face_ratio — all in dict, no rgb needed.
    # Build "shallow" IdAttrs for pool filtering.
    shallow = []
    for k, (d, _c) in enumerate(_all_attrs):
        if d["name"] == i_attr.name:
            continue
        shallow.append((k, IdAttr(
            name=d["name"], rgb=None, lm=None,
            emb=np.array(d["emb"], dtype=np.float32),
            skin_lab=np.array(d["skin_lab"], dtype=np.float32),
            face_w=d["face_w"], face_h=d["face_h"],
            face_ratio=d["face_ratio"], nose_rel=d["nose_rel"],
            iod_face_ratio=d["iod_face_ratio"],
            cheek_smoothness=d["cheek_smoothness"],
            lwr_upr_ratio=d["lwr_upr_ratio"],
        )))
    rng = random.Random(seed + hash(i_attr.name) % 2**31)
    # We need rgb+lm only on the donors actually picked. partwhole_one_identity
    # expects IdAttrs WITH rgb/lm filled in — so we adapt:
    # Filter pool, then on selected donor, load rgb/lm.
    # Easiest: pass a wrapper that lazy-loads on attribute access.
    # Simpler: pre-load rgb/lm of all candidates in the filtered pool only.
    from stimuli.classical_cv.donor_match import candidate_pool

    flat = [s for _k, s in shallow]
    pool_shallow = candidate_pool(i_attr, flat)
    if not pool_shallow:
        # Age constraints (last two args) stay STRICT — never relax
        for relax in [(0.50, 0.90, 11.0, 0.15, 0.05, 0.30),
                      (0.45, 0.95, 14.0, 0.20, 0.05, 0.30)]:
            pool_shallow = candidate_pool(i_attr, flat, *relax)
            if pool_shallow:
                break
    if not pool_shallow:
        return {"name": i_attr.name, "status": "no_pool"}
    # Load rgb/lm for pool candidates only
    name_to_idx = {d["name"]: k for k, (d, _c) in enumerate(_all_attrs)}
    full_pool: list[IdAttr] = []
    for s in pool_shallow:
        k = name_to_idx[s.name]
        full_pool.append(_load_attr_with_rgb(k))
    # Run partwhole gen on this identity, restricting all_attrs to full_pool
    result = partwhole_one_identity(i_attr, full_pool, rng)
    if result is None:
        return {"name": i_attr.name, "status": "no_donors"}
    # Save 12 PNGs
    idir.mkdir(parents=True, exist_ok=True)
    donors_meta = {}
    for feat in ["eye", "nose", "mouth"]:
        r = result[feat]
        for vk in ["V1", "V2", "V3", "V4"]:
            Image.fromarray(r[vk]).save(idir / f"partwhole_{feat}_{vk}.png")
        donors_meta[feat] = {
            "donor": r["donor_name"],
            "ssim_v1v2_bbox": r["meta"]["ssim_v1v2_bbox"],
        }
    (idir / "donors.json").write_text(json.dumps(donors_meta, indent=2))
    return {"name": i_attr.name, "status": "ok", "donors": donors_meta}


def run_phase_gen(args):
    print(f"[phase B] loading attrs from {args.attrs_pkl}")
    with open(args.attrs_pkl, "rb") as f:
        attrs_raw = pickle.load(f)
    n = len(attrs_raw)
    print(f"[phase B] {n} identities ready")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args_dict = {"output_dir": args.output_dir, "seed": args.seed}
    t0 = time.monotonic()
    counts = {"ok": 0, "skip_exists": 0, "no_pool": 0, "no_donors": 0}
    with mp.Pool(processes=args.workers,
                  initializer=_gen_init,
                  initargs=(args.attrs_pkl, args_dict)) as pool:
        for i, r in enumerate(pool.imap_unordered(
                _process_one_identity, list(range(n)))):
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if (i + 1) % 200 == 0:
                el = time.monotonic() - t0
                print(f"  [{i+1}/{n}] ok={counts['ok']} skip={counts['skip_exists']} "
                      f"no_pool={counts['no_pool']} no_donor={counts['no_donors']} "
                      f"elapsed={el:.0f}s rate={(i+1)/max(1,el):.1f}/s",
                      flush=True)
    print(f"\n[phase B] DONE: {counts}")
    print(f"  elapsed: {(time.monotonic()-t0)/60:.1f} min")


# === main ====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["attrs", "gen", "both"], default="both")
    p.add_argument("--predictor",
                   default="/workspace/models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--recognizer",
                   default="/workspace/models/dlib_face_recognition_resnet_model_v1.dat")
    p.add_argument("--source_dir",
                   default="data/thatcher_ffhq_70k_parallel")
    p.add_argument("--output_dir",
                   default="data/partwhole_ffhq_70k_parallel")
    p.add_argument("--attrs_pkl", default="data/partwhole_attrs.pkl")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260526)
    return p.parse_args()


def main():
    args = parse_args()
    if args.phase in ("attrs", "both"):
        run_phase_attrs(args)
    if args.phase in ("gen", "both"):
        run_phase_gen(args)


if __name__ == "__main__":
    main()
