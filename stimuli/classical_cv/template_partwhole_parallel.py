"""stimuli/classical_cv/template_partwhole_parallel.py — multiprocess
template-based Tanaka 2004 PW generation.

Reuses Phase A (attribute extraction) from `partwhole_parallel`. Adds new
Phase B (template generation):
  - Centralized k-means clustering picks N_TEMPLATES templates + donor pools
  - Pool.imap_unordered over templates → each worker generates one
    template's full set of (n_targets × 3 features × 4 conditions) PNGs

Output:
  data/template_partwhole_v1/
    template_<NAME>/
      target00_eye_{V1,V2,V3,V4}.png
      target00_nose_{V1,V2,V3,V4}.png
      target00_mouth_{V1,V2,V3,V4}.png
      target01_*_*.png
      ...
      manifest.json   (per-template metadata: donors, SSIM)
    manifest.csv      (global manifest)
    templates.json    (template list + cluster centroids)

Usage on server (64 cores):
  python -m stimuli.classical_cv.template_partwhole_parallel \\
    --phase=both --workers=64 --n_templates=400 \\
    --donors_per_template=15 --n_targets=6

Estimated runtime on 26k FFHQ:
  Phase A: 10-15 min (per-id dlib resnet ≈ 30-50ms × 26k / 64 workers)
  Phase B: 15-30 min (400 templates × 72 PNGs × ~0.5s each / 64 workers)
"""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stimuli.classical_cv.face_embedding import FaceEmbedder
from stimuli.classical_cv.donor_match import build_attr, IdAttr, is_likely_infant
from stimuli.classical_cv.template_partwhole import (
    build_template_stimuli, FEATURES,
)


# === Phase A (reused from partwhole_parallel, replicated for self-containment) ==

_embedder: FaceEmbedder | None = None
_args: dict | None = None


def _attrs_init(predictor_path: str, recognizer_path: str, args_dict: dict):
    global _embedder, _args
    _embedder = FaceEmbedder(predictor_path, recognizer_path)
    _args = args_dict


def _extract_attrs_for_id(id_dir_str: str) -> dict | None:
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
    print(f"[phase A] {len(id_dirs)} identity dirs", flush=True)
    t0 = time.monotonic()
    results: list[dict] = []
    with mp.Pool(processes=args.workers,
                  initializer=_attrs_init,
                  initargs=(args.predictor, args.recognizer, {})) as pool:
        for i, r in enumerate(pool.imap_unordered(
                _extract_attrs_for_id, [str(p) for p in id_dirs])):
            if r is not None:
                results.append(r)
            if (i + 1) % 500 == 0:
                el = time.monotonic() - t0
                print(f"  [{i+1}/{len(id_dirs)}] kept={len(results)} "
                      f"elapsed={el:.0f}s", flush=True)
    pre_filter = len(results)
    results = [r for r in results if not r.get("is_infant", False)]
    attrs_path = Path(args.attrs_pkl)
    attrs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(attrs_path, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"[phase A] DONE: kept {len(results)}/{len(id_dirs)} adult identities "
          f"(filtered {pre_filter - len(results)} infants)")
    print(f"  saved: {attrs_path}  ({attrs_path.stat().st_size/1e6:.1f} MB)")
    print(f"  elapsed: {(time.monotonic()-t0)/60:.1f} min")


# === Phase B: template generation =============================================

_attrs_dicts: list[dict] | None = None      # list of attr dicts (no rgb)
_template_groups: list | None = None         # set in worker init


def _dict_to_idattr(d: dict, rgb: np.ndarray, lm: np.ndarray) -> IdAttr:
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


def _load_attr_with_rgb(d: dict) -> IdAttr:
    id_dir = Path(d["id_dir"])
    rgb = np.array(Image.open(id_dir / "thatcher_V1.png").convert("RGB"))
    lm = np.array(json.loads((id_dir / "landmarks.json").read_text())["landmarks"],
                  dtype=np.float32)
    return _dict_to_idattr(d, rgb, lm)


def _gen_init(attrs_pkl: str, template_assignments_path: str,
              args_dict: dict):
    """Each Phase B worker loads attrs.pkl (small dict list) + template
    assignments (which donors belong to which template). Worker is given
    a template index, looks up its donor pool, loads RGB on demand."""
    global _attrs_dicts, _template_groups, _args
    _args = args_dict
    with open(attrs_pkl, "rb") as f:
        _attrs_dicts = pickle.load(f)
    with open(template_assignments_path, "r") as f:
        _template_groups = json.load(f)  # list of {template_name, donor_names}


def _process_one_template(template_idx: int) -> dict:
    """Worker function: generate all PW stimuli for one template."""
    global _attrs_dicts, _template_groups, _args
    out_root = Path(_args["output_dir"])
    seed = _args["seed"]
    n_targets = _args["n_targets"]
    assignment = _template_groups[template_idx]
    template_name = assignment["template_name"]
    donor_names = assignment["donor_names"]
    # Find attr dicts
    name_to_dict = {d["name"]: d for d in _attrs_dicts}
    if template_name not in name_to_dict:
        return {"template": template_name, "status": "template_missing"}
    template_dict = name_to_dict[template_name]
    template_attr = _load_attr_with_rgb(template_dict)
    donor_pool = []
    for dn in donor_names:
        if dn in name_to_dict:
            donor_pool.append(_load_attr_with_rgb(name_to_dict[dn]))
    if len(donor_pool) < 6:
        return {"template": template_name, "status": "pool_too_small",
                "pool_size": len(donor_pool)}
    # Idempotent skip: if last expected file exists, assume done
    tdir = out_root / f"template_{template_name}"
    sentinel = tdir / f"target{n_targets-1:02d}_mouth_V4.png"
    if sentinel.exists():
        return {"template": template_name, "status": "skip_exists"}
    rng = random.Random(seed + hash(template_name) % 2**31)
    try:
        results = build_template_stimuli(template_attr, donor_pool, rng,
                                          n_targets=n_targets)
    except Exception as e:
        return {"template": template_name, "status": "exception",
                "error": repr(e)}
    if not results:
        return {"template": template_name, "status": "no_results"}
    tdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        base = f"target{r['target_idx']:02d}_{r['feature']}"
        for vk in ["V1", "V2", "V3", "V4"]:
            p = tdir / f"{base}_{vk}.png"
            Image.fromarray(r[vk]).save(p)
            rows.append({
                "template": template_name,
                "target_idx": r["target_idx"],
                "feature": r["feature"],
                "condition": vk,
                "target_donor": r["target_donor_name"],
                "foil_donor": r["foil_donor_name"],
                "ssim_v1v2": r["ssim_v1v2"],
            })
    (tdir / "manifest.json").write_text(json.dumps(rows, indent=2))
    return {"template": template_name, "status": "ok", "n_pngs": len(rows)}


def _kmeans_pick_templates(attrs: list[dict], n_templates: int,
                            donors_per_template: int, seed: int
                            ) -> list[dict]:
    """Centralized k-means on attr features. Returns list of
    {template_name, donor_names, cluster_idx}."""
    feats = np.array([
        [d["skin_lab"][0], d["skin_lab"][1], d["skin_lab"][2],
         d["face_ratio"], d["lwr_upr_ratio"], d["nose_rel"]]
        for d in attrs
    ], dtype=np.float32)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    K = min(n_templates, len(attrs) // 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(
        feats, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.flatten()
    rng = random.Random(seed)
    out = []
    for k in range(K):
        members = np.where(labels == k)[0]
        if len(members) < 7:
            continue
        dists = np.linalg.norm(feats[members] - centers[k], axis=1)
        template_idx = int(members[np.argmin(dists)])
        donor_indices = [int(i) for i in members if i != template_idx]
        rng.shuffle(donor_indices)
        donor_indices = donor_indices[:donors_per_template]
        out.append({
            "template_name": attrs[template_idx]["name"],
            "donor_names": [attrs[i]["name"] for i in donor_indices],
            "cluster_idx": k,
        })
    return out


def run_phase_gen(args):
    print(f"[phase B] loading attrs from {args.attrs_pkl}")
    with open(args.attrs_pkl, "rb") as f:
        attrs = pickle.load(f)
    print(f"[phase B] {len(attrs)} adult attrs loaded")
    print(f"[phase B] k-means clustering into {args.n_templates} templates...")
    template_groups = _kmeans_pick_templates(
        attrs, args.n_templates, args.donors_per_template, args.seed)
    print(f"[phase B] {len(template_groups)} templates picked")
    assignments_path = Path(args.attrs_pkl).parent / "templates.json"
    assignments_path.write_text(json.dumps(template_groups, indent=2))
    print(f"[phase B] saved template assignments: {assignments_path}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args_dict = {"output_dir": args.output_dir, "seed": args.seed,
                 "n_targets": args.n_targets}
    t0 = time.monotonic()
    counts = {"ok": 0, "skip_exists": 0, "no_results": 0,
              "pool_too_small": 0, "template_missing": 0, "exception": 0}
    total_pngs = 0
    with mp.Pool(processes=args.workers,
                  initializer=_gen_init,
                  initargs=(args.attrs_pkl, str(assignments_path),
                            args_dict)) as pool:
        for i, r in enumerate(pool.imap_unordered(
                _process_one_template, list(range(len(template_groups))))):
            status = r.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
            total_pngs += r.get("n_pngs", 0)
            if (i + 1) % 20 == 0 or status == "exception":
                el = time.monotonic() - t0
                print(f"  [{i+1}/{len(template_groups)}] ok={counts['ok']} "
                      f"pngs={total_pngs} skip={counts['skip_exists']} "
                      f"elapsed={el:.0f}s", flush=True)
                if status == "exception":
                    print(f"    !! {r.get('template')}: {r.get('error')}")
    print(f"\n[phase B] DONE: {counts}")
    print(f"  total PNGs: {total_pngs}")
    print(f"  elapsed: {(time.monotonic()-t0)/60:.1f} min")
    # Aggregate manifest across templates
    out_root = Path(args.output_dir)
    global_manifest = []
    for tdir in sorted(out_root.glob("template_*")):
        mp_json = tdir / "manifest.json"
        if mp_json.exists():
            global_manifest.extend(json.loads(mp_json.read_text()))
    if global_manifest:
        mpath = out_root / "manifest.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(global_manifest[0].keys()))
            w.writeheader(); w.writerows(global_manifest)
        print(f"  global manifest: {mpath}  ({len(global_manifest)} rows)")


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
                   default="data/template_partwhole_v1")
    p.add_argument("--attrs_pkl", default="data/template_partwhole_attrs.pkl")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--n_templates", type=int, default=400)
    p.add_argument("--donors_per_template", type=int, default=15)
    p.add_argument("--n_targets", type=int, default=6)
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
