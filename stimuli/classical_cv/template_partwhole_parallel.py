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
from stimuli.classical_cv.donor_match import (
    build_attr, IdAttr, is_likely_infant, is_blinking,
)
from stimuli.classical_cv.template_partwhole import (
    build_template_stimuli, FEATURES,
)


# === Phase A (reused from partwhole_parallel, replicated for self-containment) ==

_embedder: FaceEmbedder | None = None
_insight = None         # InsightAttrExtractor instance per worker
_args: dict | None = None


def _attrs_init(predictor_path: str, recognizer_path: str,
                use_insight: bool, args_dict: dict):
    global _embedder, _insight, _args
    _embedder = FaceEmbedder(predictor_path, recognizer_path)
    _args = args_dict
    if use_insight:
        from stimuli.classical_cv.insight_attrs import InsightAttrExtractor
        _insight = InsightAttrExtractor()


def _extract_attrs_for_id(id_dir_str: str) -> dict | None:
    global _embedder, _insight
    id_dir = Path(id_dir_str)
    lm_path = id_dir / "landmarks.json"
    img_path = id_dir / "thatcher_V1.png"
    if not (lm_path.exists() and img_path.exists()):
        return None
    try:
        meta = json.loads(lm_path.read_text())
        rgb = np.array(Image.open(img_path).convert("RGB"))
        lm = np.array(meta["landmarks"], dtype=np.float32)
        attr = build_attr(meta["ffhq_name"], rgb, lm, _embedder,
                          age_gender_predictor=_insight)
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
            "eye_width_iod":     attr.eye_width_iod,
            "eye_ear":           attr.eye_ear,
            "nose_height_face":  attr.nose_height_face,
            "mouth_width_face":  attr.mouth_width_face,
            "predicted_age":     attr.predicted_age,
            "predicted_gender":  attr.predicted_gender,
            "is_infant":         is_likely_infant(attr),
            "is_blinking":       is_blinking(attr),
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
    use_insight = bool(getattr(args, "use_insight", False))
    print(f"[phase A] use_insight={use_insight} workers={args.workers}", flush=True)
    with mp.Pool(processes=args.workers,
                  initializer=_attrs_init,
                  initargs=(args.predictor, args.recognizer, use_insight, {})) as pool:
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
        eye_width_iod=d.get("eye_width_iod", 0.0),
        eye_ear=d.get("eye_ear", 0.0),
        nose_height_face=d.get("nose_height_face", 0.0),
        mouth_width_face=d.get("mouth_width_face", 0.0),
        predicted_age=d.get("predicted_age", -1),
        predicted_gender=d.get("predicted_gender", "?"),
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
    """Worker function: generate all PW stimuli for one template.

    Supports BOTH legacy format ({"donor_names": [...]}) and per-feature
    format ({"donors_per_feature": {"eye":[...], ...}}). Per-feature
    routing skips any feature whose pool is empty (template-feature pair
    rejected by kNN filter)."""
    global _attrs_dicts, _template_groups, _args
    out_root = Path(_args["output_dir"])
    seed = _args["seed"]
    n_targets = _args["n_targets"]
    assignment = _template_groups[template_idx]
    template_name = assignment["template_name"]
    name_to_dict = {d["name"]: d for d in _attrs_dicts}
    if template_name not in name_to_dict:
        return {"template": template_name, "status": "template_missing"}
    template_dict = name_to_dict[template_name]
    template_attr = _load_attr_with_rgb(template_dict)
    # Build donor pools — per-feature if available, else legacy
    if "donors_per_feature" in assignment:
        donor_pools_per_feature: dict[str, list[IdAttr]] = {}
        for feat in ("eye", "nose", "mouth"):
            names = assignment["donors_per_feature"].get(feat, [])
            donor_pools_per_feature[feat] = [
                _load_attr_with_rgb(name_to_dict[n]) for n in names
                if n in name_to_dict
            ]
        active = [f for f, p in donor_pools_per_feature.items()
                  if len(p) >= n_targets]
        if not active:
            return {"template": template_name, "status": "pool_too_small",
                    "active_features": []}
    else:
        donor_pool = []
        for dn in assignment.get("donor_names", []):
            if dn in name_to_dict:
                donor_pool.append(_load_attr_with_rgb(name_to_dict[dn]))
        if len(donor_pool) < 6:
            return {"template": template_name, "status": "pool_too_small",
                    "pool_size": len(donor_pool)}
        donor_pools_per_feature = None
    # Idempotent skip: if last expected file exists, assume done
    tdir = out_root / f"template_{template_name}"
    sentinel = tdir / f"target{n_targets-1:02d}_mouth_V4.png"
    if sentinel.exists():
        return {"template": template_name, "status": "skip_exists"}
    rng = random.Random(seed + hash(template_name) % 2**31)
    try:
        if donor_pools_per_feature is not None:
            results = build_template_stimuli(
                template_attr, donor_pools_per_feature, rng,
                n_targets=n_targets, per_feature_pools=True)
        else:
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


def _knn_pick_templates_per_feature(
        attrs: list[dict], donors_per_template: int,
        query_k: int = 500,
        # Template-level demographic constraints (must pass to be candidate at all)
        lab_L_max: float = 8.0,
        lab_a_max: float = 6.0,
        lab_b_max: float = 6.0,
        face_ratio_max: float = 0.15,
        iod_ratio_max: float = 0.05,
        smoothness_log_max: float = 0.7,
        d_emb_low: float = 0.60,
        d_emb_high: float = 0.85,
        # InsightFace gender/age constraints (skipped if predictions unavailable)
        require_gender_match: bool = True,
        age_diff_max: int = 15,
        # Per-feature size constraints
        eye_size_diff_max:   float = 0.10,   # |Δ eye_width_iod| (typical ~ 0.45-0.55)
        nose_size_diff_max:  float = 0.025,  # |Δ nose_width_face|; nose_w_face ~ 0.18-0.25
        mouth_size_diff_max: float = 0.04,   # |Δ mouth_width_face|; ~ 0.28-0.38
        # "If both are LARGE on this feature → skip swap" — Pxx defined per feature
        skip_both_large_pct: float = 75.0,   # P75 cutoff
        min_donors_per_feature: int = 6,
        ) -> list[dict]:
    """Per-feature KDTree donor selection.

    Two-stage filter:
      Stage 1 (template-level): demographic hard constraints — gender match,
        age diff ≤ 15, skin LAB Δ ≤ 8/6/6, face geometry, embedding distance.
      Stage 2 (per-feature):    for each of {eye, nose, mouth} independently:
        - |size_template - size_donor| <= eps_feature
        - skip if BOTH template and donor's feature size > P75 across dataset
          (user request 2026-05-27: "if both have very large feature, don't
           swap" — large features create distinct stimuli that break the
           subtle PW illusion)

    Returns per-template per-feature donor lists. Template is DROPPED if any
    feature's pool has fewer than min_donors_per_feature passing candidates.
    """
    from sklearn.neighbors import KDTree

    feats = np.array([
        [d["skin_lab"][0], d["skin_lab"][1], d["skin_lab"][2],
         d["face_ratio"], d["lwr_upr_ratio"], d["nose_rel"]]
        for d in attrs
    ], dtype=np.float64)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    embs = np.array([d["emb"] for d in attrs], dtype=np.float32)  # (N, 128)

    # P75 cutoffs per feature (computed across population)
    eye_w_arr   = np.array([d.get("eye_width_iod",   0.5) for d in attrs])
    nose_w_arr  = np.array([d.get("nose_rel",        0.2) for d in attrs])
    mouth_w_arr = np.array([d.get("mouth_width_face", 0.3) for d in attrs])
    eye_w_p75   = float(np.percentile(eye_w_arr,   skip_both_large_pct))
    nose_w_p75  = float(np.percentile(nose_w_arr,  skip_both_large_pct))
    mouth_w_p75 = float(np.percentile(mouth_w_arr, skip_both_large_pct))
    print(f"[knn-filter] P{skip_both_large_pct:.0f} cutoffs: "
          f"eye_w_iod={eye_w_p75:.3f}  nose_rel={nose_w_p75:.3f}  "
          f"mouth_w_face={mouth_w_p75:.3f}", flush=True)

    k = min(len(attrs), query_k)
    tree = KDTree(feats)
    indices = tree.query(feats, k=k, return_distance=False)

    out = []
    n_dropped_no_demo = 0
    n_dropped_per_feature = {f: 0 for f in ("eye", "nose", "mouth")}
    n_dropped_template = 0
    for i in range(len(attrs)):
        di = attrs[i]
        ls_i = float(np.log10(di["cheek_smoothness"] + 1.0))
        gi = di.get("predicted_gender", "?")
        ai = di.get("predicted_age", -1)
        # Stage 1: template-level filter → list of valid candidate idxs
        demo_pool: list[int] = []
        for j in indices[i]:
            if j == i:
                continue
            dj = attrs[int(j)]
            # Gender hard match (only enforce if BOTH have predictions)
            if require_gender_match:
                gj = dj.get("predicted_gender", "?")
                if gi != "?" and gj != "?" and gi != gj:
                    continue
            # Age diff
            aj = dj.get("predicted_age", -1)
            if ai >= 0 and aj >= 0 and abs(ai - aj) > age_diff_max:
                continue
            # Skin LAB
            if abs(di["skin_lab"][0] - dj["skin_lab"][0]) > lab_L_max:
                continue
            if abs(di["skin_lab"][1] - dj["skin_lab"][1]) > lab_a_max:
                continue
            if abs(di["skin_lab"][2] - dj["skin_lab"][2]) > lab_b_max:
                continue
            # Face geometry
            if abs(di["face_ratio"] - dj["face_ratio"]) > face_ratio_max:
                continue
            if abs(di["iod_face_ratio"] - dj["iod_face_ratio"]) > iod_ratio_max:
                continue
            # Smoothness (age proxy)
            ls_j = float(np.log10(dj["cheek_smoothness"] + 1.0))
            if abs(ls_i - ls_j) > smoothness_log_max:
                continue
            # Embedding distance
            d_emb = float(np.linalg.norm(embs[i] - embs[int(j)]))
            if not (d_emb_low <= d_emb <= d_emb_high):
                continue
            demo_pool.append(int(j))
        if len(demo_pool) < min_donors_per_feature:
            n_dropped_no_demo += 1
            continue
        # Stage 2: per-feature filter on demo_pool
        eye_pool, nose_pool, mouth_pool = [], [], []
        tpl_eye_w   = di.get("eye_width_iod",   0.5)
        tpl_nose_w  = di.get("nose_rel",        0.2)
        tpl_mouth_w = di.get("mouth_width_face", 0.3)
        tpl_blinking = di.get("is_blinking", False)
        for j in demo_pool:
            dj = attrs[j]
            # EYE pool
            if len(eye_pool) < donors_per_template:
                ok = True
                if dj.get("is_blinking", False):
                    ok = False  # never use blinking donor for eye swap
                d_size = abs(tpl_eye_w - dj.get("eye_width_iod", 0.5))
                if d_size > eye_size_diff_max:
                    ok = False
                # Skip if both large
                if (tpl_eye_w > eye_w_p75 and
                        dj.get("eye_width_iod", 0.5) > eye_w_p75):
                    ok = False
                if ok:
                    eye_pool.append(j)
            # NOSE pool
            if len(nose_pool) < donors_per_template:
                ok = True
                d_size = abs(tpl_nose_w - dj.get("nose_rel", 0.2))
                if d_size > nose_size_diff_max:
                    ok = False
                if (tpl_nose_w > nose_w_p75 and
                        dj.get("nose_rel", 0.2) > nose_w_p75):
                    ok = False
                if ok:
                    nose_pool.append(j)
            # MOUTH pool
            if len(mouth_pool) < donors_per_template:
                ok = True
                d_size = abs(tpl_mouth_w - dj.get("mouth_width_face", 0.3))
                if d_size > mouth_size_diff_max:
                    ok = False
                if (tpl_mouth_w > mouth_w_p75 and
                        dj.get("mouth_width_face", 0.3) > mouth_w_p75):
                    ok = False
                if ok:
                    mouth_pool.append(j)
        # If template is blinking, drop its eye pool (don't generate eye swaps)
        if tpl_blinking:
            eye_pool = []
        # Check pool sizes per feature
        pool_sizes = {"eye": len(eye_pool), "nose": len(nose_pool),
                      "mouth": len(mouth_pool)}
        kept_features = {f for f, sz in pool_sizes.items()
                         if sz >= min_donors_per_feature}
        for f, sz in pool_sizes.items():
            if sz < min_donors_per_feature:
                n_dropped_per_feature[f] += 1
        if not kept_features:
            n_dropped_template += 1
            continue
        out.append({
            "template_name": di["name"],
            "donors_per_feature": {
                "eye":   [attrs[j]["name"] for j in eye_pool]   if "eye"   in kept_features else [],
                "nose":  [attrs[j]["name"] for j in nose_pool]  if "nose"  in kept_features else [],
                "mouth": [attrs[j]["name"] for j in mouth_pool] if "mouth" in kept_features else [],
            },
        })
    print(f"[knn-filter] yield: kept {len(out)} / {len(attrs)} templates", flush=True)
    print(f"  dropped (no demo donors):  {n_dropped_no_demo}", flush=True)
    print(f"  dropped per-feature pool too small: {n_dropped_per_feature}", flush=True)
    print(f"  dropped (no features kept): {n_dropped_template}", flush=True)
    n_eye   = sum(1 for r in out if r["donors_per_feature"]["eye"])
    n_nose  = sum(1 for r in out if r["donors_per_feature"]["nose"])
    n_mouth = sum(1 for r in out if r["donors_per_feature"]["mouth"])
    print(f"  templates with eye/nose/mouth pool: {n_eye}/{n_nose}/{n_mouth}", flush=True)
    return out


def _knn_pick_templates(attrs: list[dict], donors_per_template: int,
                         **kwargs) -> list[dict]:
    """Back-compat alias. Calls per-feature picker."""
    return _knn_pick_templates_per_feature(attrs, donors_per_template, **kwargs)


def run_phase_gen(args):
    print(f"[phase B] loading attrs from {args.attrs_pkl}")
    with open(args.attrs_pkl, "rb") as f:
        attrs = pickle.load(f)
    print(f"[phase B] {len(attrs)} adult attrs loaded")

    mode = getattr(args, "template_mode", "kmeans")
    if mode == "knn":
        print(f"[phase B] kNN templates: {len(attrs)} templates "
              f"× {args.donors_per_template} donors each ...")
        template_groups = _knn_pick_templates(attrs, args.donors_per_template)
    else:
        print(f"[phase B] k-means clustering into {args.n_templates} templates...")
        template_groups = _kmeans_pick_templates(
            attrs, args.n_templates, args.donors_per_template, args.seed)
    print(f"[phase B] {len(template_groups)} templates picked")
    # Optional subsample for dry-run validation
    max_t = getattr(args, "max_templates", 0)
    if max_t and len(template_groups) > max_t:
        rng_sub = random.Random(args.seed + 1)
        rng_sub.shuffle(template_groups)
        template_groups = template_groups[:max_t]
        print(f"[phase B] subsampled to {len(template_groups)} templates "
              f"(--max_templates={max_t})")
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
    p.add_argument("--template_mode", choices=["kmeans", "knn"], default="kmeans",
                   help="kmeans: cluster centroids as templates; "
                        "knn: every adult face is a template with k NN donors")
    p.add_argument("--max_templates", type=int, default=0,
                   help="If > 0, subsample template_groups to this many for dry-run")
    p.add_argument("--use_insight", action="store_true",
                   help="Use InsightFace for age/gender prediction in Phase A")
    return p.parse_args()


def main():
    args = parse_args()
    if args.phase in ("attrs", "both"):
        run_phase_attrs(args)
    if args.phase in ("gen", "both"):
        run_phase_gen(args)


if __name__ == "__main__":
    main()
