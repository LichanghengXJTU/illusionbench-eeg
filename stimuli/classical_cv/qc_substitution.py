"""stimuli/classical_cv/qc_substitution.py — verify that PW feature swaps
are perceptually visible.

For each (template, target, feature) row in a generated output dir, computes
in the feature's TIGHT bbox (region where V3 differs from gray background):

  SSIM(V1, V2)
  mean|V1 - V2|  (per pixel, 0..255)
  %pixels with |V1 - V2| > 10
  same metrics on V3 vs V4 (isolated parts should also differ)

Pass criteria — a substitution that's "perceptually visible":
  V1 vs V2 in feature bbox:
    SSIM in [0.40, 0.75]
      < 0.40 → too jarring (donor mismatch); >0.75 → swap invisible
    mean|diff| >= 5  (donor content changes >2% of luminance range)
    %changed >= 10%  (donor content occupies >10% of feature region)

Usage:
  python -m stimuli.classical_cv.qc_substitution \\
    --output_dir data/template_partwhole_v3_dryrun \\
    --report     data/template_partwhole_v3_dryrun/qc_report.csv

Output:
  - qc_report.csv   (one row per (template, target, feature))
  - qc_summary.json (pass rates per feature, distribution stats)
  - prints summary to stdout
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


GRAY_VALUE = 128


def feature_bbox_from_v3(v3: np.ndarray, gray: int = GRAY_VALUE,
                          tol: int = 2) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) of the V3 non-gray region (i.e. the tight
    feature polygon). Returns None if V3 is all gray."""
    diff = np.any(np.abs(v3.astype(np.int32) - gray) > tol, axis=-1)
    if not diff.any():
        return None
    ys, xs = np.where(diff)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def metrics_in_bbox(a: np.ndarray, b: np.ndarray,
                     bbox: tuple[int, int, int, int],
                     diff_thresh: int = 10) -> dict:
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 4 or y2 - y1 < 4:
        return {"ssim": float("nan"), "mean_diff": float("nan"),
                "pct_changed": float("nan"), "bbox_area": 0}
    a_crop = a[y1:y2, x1:x2]
    b_crop = b[y1:y2, x1:x2]
    diff = np.abs(a_crop.astype(np.int32) - b_crop.astype(np.int32))
    pixel_diff = diff.sum(-1)  # sum over RGB
    s = float(ssim(a_crop, b_crop, channel_axis=2, data_range=255))
    return {
        "ssim": s,
        "mean_diff": float(diff.mean()),
        "pct_changed": float((pixel_diff > diff_thresh).mean() * 100),
        "bbox_area": int((x2 - x1) * (y2 - y1)),
    }


def is_passing(m: dict,
                ssim_low: float = 0.40, ssim_high: float = 0.75,
                mean_diff_min: float = 5.0,
                pct_changed_min: float = 10.0) -> tuple[bool, str]:
    """Return (pass, reason). reason is empty on pass."""
    if not np.isfinite(m["ssim"]):
        return False, "bbox_too_small"
    if m["ssim"] > ssim_high:
        return False, f"ssim_too_high({m['ssim']:.2f}>{ssim_high})"
    if m["ssim"] < ssim_low:
        return False, f"ssim_too_low({m['ssim']:.2f}<{ssim_low})"
    if m["mean_diff"] < mean_diff_min:
        return False, f"mean_diff_too_low({m['mean_diff']:.1f}<{mean_diff_min})"
    if m["pct_changed"] < pct_changed_min:
        return False, f"pct_changed_too_low({m['pct_changed']:.1f}<{pct_changed_min})"
    return True, ""


def process_template_dir(tdir: Path) -> list[dict]:
    """Process one template_NNNNN/ directory; return list of metric rows."""
    mf_path = tdir / "manifest.json"
    if not mf_path.exists():
        return []
    rows_meta = json.loads(mf_path.read_text())
    # Group by (target_idx, feature): each group has V1, V2, V3, V4
    keys = {}
    for r in rows_meta:
        k = (r["target_idx"], r["feature"])
        keys.setdefault(k, {})[r["condition"]] = r
    out_rows = []
    for (t_idx, feat), conds in sorted(keys.items()):
        if not all(c in conds for c in ("V1", "V2", "V3", "V4")):
            out_rows.append({
                "template": tdir.name.replace("template_", ""),
                "target_idx": t_idx, "feature": feat,
                "status": "missing_conditions",
            })
            continue
        try:
            v1 = np.array(Image.open(tdir / f"target{t_idx:02d}_{feat}_V1.png"))
            v2 = np.array(Image.open(tdir / f"target{t_idx:02d}_{feat}_V2.png"))
            v3 = np.array(Image.open(tdir / f"target{t_idx:02d}_{feat}_V3.png"))
            v4 = np.array(Image.open(tdir / f"target{t_idx:02d}_{feat}_V4.png"))
        except Exception as e:
            out_rows.append({
                "template": tdir.name.replace("template_", ""),
                "target_idx": t_idx, "feature": feat,
                "status": f"read_error:{e}",
            })
            continue
        bbox = feature_bbox_from_v3(v3)
        if bbox is None:
            out_rows.append({
                "template": tdir.name.replace("template_", ""),
                "target_idx": t_idx, "feature": feat,
                "status": "v3_all_gray",
            })
            continue
        m_whole = metrics_in_bbox(v1, v2, bbox)
        m_part  = metrics_in_bbox(v3, v4, bbox)
        pass_whole, reason_whole = is_passing(m_whole)
        pass_part,  reason_part  = is_passing(m_part)
        meta_first = conds["V1"]
        out_rows.append({
            "template": tdir.name.replace("template_", ""),
            "target_idx": t_idx,
            "feature": feat,
            "target_donor": meta_first.get("target_donor", ""),
            "foil_donor":   meta_first.get("foil_donor", ""),
            "bbox_area":    m_whole["bbox_area"],
            "whole_ssim":        m_whole["ssim"],
            "whole_mean_diff":   m_whole["mean_diff"],
            "whole_pct_changed": m_whole["pct_changed"],
            "whole_pass":        pass_whole,
            "whole_reason":      reason_whole,
            "part_ssim":         m_part["ssim"],
            "part_mean_diff":    m_part["mean_diff"],
            "part_pct_changed":  m_part["pct_changed"],
            "part_pass":         pass_part,
            "part_reason":       reason_part,
            "status": "ok",
        })
    return out_rows


def summarize(rows: list[dict]) -> dict:
    """Per-feature pass rates and metric distributions."""
    by_feat = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        f = r["feature"]
        by_feat.setdefault(f, []).append(r)
    summary = {"n_total": len(rows), "by_feature": {}}
    for f, fr in by_feat.items():
        n = len(fr)
        whole_pass = sum(1 for r in fr if r["whole_pass"])
        part_pass  = sum(1 for r in fr if r["part_pass"])
        both_pass  = sum(1 for r in fr if r["whole_pass"] and r["part_pass"])
        def stats(key):
            arr = np.array([r[key] for r in fr if np.isfinite(r[key])])
            if not len(arr):
                return None
            return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "p90": float(np.percentile(arr, 90))}
        summary["by_feature"][f] = {
            "n": n,
            "whole_pass_rate": whole_pass / n,
            "part_pass_rate":  part_pass / n,
            "both_pass_rate":  both_pass / n,
            "whole_ssim":        stats("whole_ssim"),
            "whole_mean_diff":   stats("whole_mean_diff"),
            "whole_pct_changed": stats("whole_pct_changed"),
            "part_ssim":         stats("part_ssim"),
            "part_mean_diff":    stats("part_mean_diff"),
            "part_pct_changed":  stats("part_pct_changed"),
        }
        # Failure reason histogram
        reasons = {}
        for r in fr:
            if not r["whole_pass"]:
                reasons[r["whole_reason"]] = reasons.get(r["whole_reason"], 0) + 1
        summary["by_feature"][f]["whole_failure_reasons"] = reasons
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="Generated dataset root (contains template_NNNNN/ dirs)")
    ap.add_argument("--report", default=None,
                    help="CSV path for per-row report (default: <output_dir>/qc_report.csv)")
    ap.add_argument("--summary", default=None,
                    help="JSON path for summary (default: <output_dir>/qc_summary.json)")
    args = ap.parse_args()
    out_root = Path(args.output_dir)
    if not out_root.exists():
        print(f"ERROR: {out_root} does not exist", file=sys.stderr); sys.exit(1)
    tdirs = sorted(out_root.glob("template_*"))
    if not tdirs:
        print(f"ERROR: no template_* dirs in {out_root}", file=sys.stderr); sys.exit(1)
    print(f"[qc] {len(tdirs)} template dirs in {out_root}", flush=True)
    all_rows = []
    for i, tdir in enumerate(tdirs):
        rows = process_template_dir(tdir)
        all_rows.extend(rows)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tdirs)}]", flush=True)
    report_path = Path(args.report) if args.report else out_root / "qc_report.csv"
    summary_path = Path(args.summary) if args.summary else out_root / "qc_summary.json"
    # Write report
    if all_rows:
        keys_union = set()
        for r in all_rows:
            keys_union.update(r.keys())
        fieldnames = sorted(keys_union)
        with open(report_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"[qc] wrote per-row report: {report_path}  ({len(all_rows)} rows)")
    summary = summarize(all_rows)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[qc] wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
