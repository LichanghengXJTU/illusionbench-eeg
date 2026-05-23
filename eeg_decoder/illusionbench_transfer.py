"""eeg_decoder/illusionbench_transfer.py — apply the per-subject EEG bottleneck
(estimated as per-dim Pearson r_d on the THINGS-EEG2 test set, from tick-83's
ridge ATM-EEG → DINOv2 fit) to the IllusionBench-EEG stimulus DINOv2 features.

Then re-run `analysis.compute_metrics` on the filtered embeddings to obtain
ISI / CSI / PWI / ISIrbox under the simulated EEG bottleneck.

Pipeline:
  Per paradigm in {thatcher, composite, partwhole, randombbox}:
    raw NPZ at `outputs/embeddings/{paradigm}_HOLONET_v2_dinov2/HOLONET_v2_dinov2_afp.npz`
    has L2-normalized DINOv2 mean-pooled patch features (shape (N, 384)).

    For each subject (and the aggregate = mean r_d over 10 subjects):
      filtered = (emb * r_d) / ||emb * r_d||           # per-dim filter + L2-renorm
      save NPZ at `outputs/embeddings/{paradigm}_EEG_dinov2_{tag}/EEG_dinov2_{tag}_afp.npz`
      run `analysis.compute_metrics` to produce a CSV row per layer

  Aggregate per-subject indices + the aggregate-r_d quartet.
  Apply pixel correction and §6 criteria.

  Verdict: per-subject distribution + aggregate quartet vs §6 thresholds.

Pre-registered predictions (filed in EEG_DECODER_STAGE3_DESIGN.md §6):
  - ISIrbox preserved (≤ 1.5) — high confidence (raw 0.654)
  - PWI preserved (≤ 0.5) — moderate (raw 0.446; attenuation could push toward
    chance 0.7-1.0)
  - CSI preserved (≥ 1.5 ideal, ≥ 1.2 partial) — low confidence (raw 1.300
    already FAIL)
  - ISI improved (≥ 1.0 = no anti-Thatcher) — low-moderate (raw 0.901)

Run from repo root:
  cd /workspace/illusionbench-eeg
  python -m eeg_decoder.illusionbench_transfer \
      --r_d_path /workspace/runs/2026-05-23_stage3-ridge_seed20260521/per_dim_preservation.pt \
      --output_root /workspace/illusionbench-eeg/outputs
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PARADIGMS = {
    "thatcher":   ("data/stimuli_ffhq/thatcher_manifest.csv",            1.000, "ISI"),
    "composite":  ("data/stimuli_ffhq_composite/thatcher_manifest.csv",  1.099, "CSI"),
    "partwhole":  ("data/stimuli_ffhq_partwhole/thatcher_manifest.csv",  1.202, "PWI"),
    "randombbox": ("data/stimuli_ffhq_randombbox/thatcher_manifest.csv", 1.000, "ISIrbox"),
}
CRITERIA = {"thatcher": (">=", 3.0), "composite": (">=", 1.5),
            "partwhole": ("<=", 0.5), "randombbox": ("<=", 1.5)}


def filter_by_rd(embeddings: np.ndarray, r_d: np.ndarray) -> np.ndarray:
    """Apply per-dim r_d filter and L2-renormalize. emb (N, D) × r_d (D,)
    → (N, D) L2-normalized."""
    f = embeddings * r_d[None, :]
    n = np.linalg.norm(f, axis=1, keepdims=True).clip(min=1e-8)
    return (f / n).astype(np.float32)


def make_filtered_npz(src_npz: Path, dst_npz: Path, r_d: np.ndarray,
                       model_id: str) -> None:
    d = np.load(src_npz, allow_pickle=True)
    emb = d["embeddings"]
    assert emb.shape[1] == r_d.shape[0], \
        f"dim mismatch: emb {emb.shape} vs r_d {r_d.shape}"
    filtered = filter_by_rd(emb, r_d)
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_npz,
             stim_ids=d["stim_ids"], embeddings=filtered,
             model_id=model_id, output_dim=int(filtered.shape[1]))


def run_compute_metrics(emb_dir: Path, manifest: Path, output_csv: Path,
                          cwd: Path) -> None:
    cmd = [sys.executable, "-m", "analysis.compute_metrics",
           "--embedding_dir", str(emb_dir),
           "--manifest", str(manifest),
           "--output", str(output_csv)]
    subprocess.run(cmd, check=True, cwd=str(cwd),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args):
    r_d_dict = torch.load(args.r_d_path, map_location="cpu", weights_only=True)
    sub_keys = sorted(r_d_dict.keys())
    print(f"[r_d] {len(sub_keys)} subjects: {sub_keys}")
    # Aggregate
    r_d_agg = torch.stack([r_d_dict[k] for k in sub_keys]).mean(dim=0)
    print(f"[r_d] aggregate mean={r_d_agg.mean().item():.3f} std={r_d_agg.std().item():.3f} "
          f"min={r_d_agg.min().item():.3f} max={r_d_agg.max().item():.3f}")

    repo = Path(args.repo_root)
    out_root = Path(args.output_root)

    # Per-subject + aggregate filter
    all_filters = {f"sub-{i:02d}": r_d_dict[f"sub-{i:02d}"].numpy()
                    for i in range(1, 11)}
    all_filters["agg"] = r_d_agg.numpy()
    # Also include raw (no filter) as control — r_d = 1 for all dims
    all_filters["raw"] = np.ones(384, dtype=np.float32)

    # For each paradigm × filter: produce filtered NPZ + run metrics
    per_para_per_filt: dict[str, dict[str, dict[str, float]]] = {}
    for para, (manifest_rel, pixbase, idx_name) in PARADIGMS.items():
        per_para_per_filt[para] = {}
        src_npz = (out_root / "embeddings" /
                   f"{para}_HOLONET_v2_dinov2" / "HOLONET_v2_dinov2_afp.npz")
        for filt_name, r_d in all_filters.items():
            tag = f"EEG_dinov2_{filt_name}"
            dst_dir = out_root / "embeddings" / f"{para}_{tag}"
            dst_npz = dst_dir / f"{tag}_afp.npz"
            make_filtered_npz(src_npz, dst_npz, r_d, f"{tag}_afp")
            table = out_root / "tables" / f"{tag}_{para}.csv"
            run_compute_metrics(dst_dir, repo / manifest_rel, table, repo)
            row = pd.read_csv(table)
            isi = float(row[row["model_id"] == f"{tag}_afp"]["isi_pert"].iloc[0])
            isi_lo = float(row[row["model_id"] == f"{tag}_afp"]["isi_pert_ci_lo"].iloc[0])
            isi_hi = float(row[row["model_id"] == f"{tag}_afp"]["isi_pert_ci_hi"].iloc[0])
            per_para_per_filt[para][filt_name] = {
                "isi_pert": isi, "ci_lo": isi_lo, "ci_hi": isi_hi,
                "pixel_baseline": pixbase,
                "pixel_corrected": isi / pixbase,
            }
        print(f"[{para}] done")

    # ---- assemble and print headline tables ----
    def fmt(v): return f"{v:7.3f}"

    print("\n=== IllusionBench-EEG transfer — per-subject ridge bottleneck ===")
    print("filter      " + "".join(f"{PARADIGMS[p][2]:>11}" for p in PARADIGMS))
    sub_order = [f"sub-{i:02d}" for i in range(1, 11)] + ["agg", "raw"]
    for filt in sub_order:
        line = f"{filt:11s}"
        for p in PARADIGMS:
            v = per_para_per_filt[p][filt]["pixel_corrected"]
            line += fmt(v)
        print(line)

    # ---- §6 verdict at the EEG-decoded (aggregate-r_d) layer ----
    print("\n=== §6 verdict at EEG-decoded DINOv2 (aggregate r_d filter) ===")
    all_pass_agg = True
    for p, (op, thr) in CRITERIA.items():
        v = per_para_per_filt[p]["agg"]["pixel_corrected"]
        v_raw = per_para_per_filt[p]["raw"]["pixel_corrected"]
        ok = (v >= thr) if op == ">=" else (v <= thr)
        all_pass_agg = all_pass_agg and ok
        print(f"  {PARADIGMS[p][2]:8s} = {v:7.3f}   need {op} {thr:<4}  "
              f"{'PASS' if ok else 'FAIL'}   (raw DINOv2 was {v_raw:.3f})")
    print(f"\n  ALL FOUR (aggregate): "
          f"{'*** PASS ***' if all_pass_agg else 'FAIL — at least one criterion not met'}")

    # ---- per-subject verdict count ----
    print("\n=== per-subject §6 pass count (of 4) ===")
    per_sub_pass = {}
    for sub in [f"sub-{i:02d}" for i in range(1, 11)]:
        n_pass = 0
        marks = []
        for p, (op, thr) in CRITERIA.items():
            v = per_para_per_filt[p][sub]["pixel_corrected"]
            ok = (v >= thr) if op == ">=" else (v <= thr)
            n_pass += int(ok)
            marks.append(f"{PARADIGMS[p][2]}={'PASS' if ok else 'FAIL'}")
        per_sub_pass[sub] = n_pass
        print(f"  {sub}: {n_pass}/4  ({', '.join(marks)})")

    # Save summary
    out_dir = Path(args.output_root) / "stage3_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "transfer_summary.json"
    summary = {
        "per_para_per_filt": {p: {f: {k: float(v) for k, v in d.items()}
                                    for f, d in fd.items()}
                                for p, fd in per_para_per_filt.items()},
        "per_sub_pass_of_4": per_sub_pass,
        "all_pass_agg": all_pass_agg,
        "config": vars(args),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {json_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--r_d_path",
                   default="/workspace/runs/2026-05-23_stage3-ridge_seed20260521/per_dim_preservation.pt")
    p.add_argument("--repo_root", default="/workspace/illusionbench-eeg")
    p.add_argument("--output_root", default="/workspace/illusionbench-eeg/outputs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
