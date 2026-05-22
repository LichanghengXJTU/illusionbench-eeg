"""holo_net/eval_falsification.py — full 4-paradigm falsification evaluation of a
trained HOLO-Net checkpoint.

For each of {thatcher, composite, part-whole, random-bbox}:
  eval_extract (per-brain-region-layer NPZ) → compute_metrics (per-layer index)
→ assemble a (layer × paradigm) table, pixel-correct, and check the FFA-layer
row against the pre-registered falsification criteria (design doc §6):

  Thatcher ISI ≥ 3.0,  Composite CSI ≥ 1.5,  Part-Whole PWI ≤ 0.5,
  Random-bbox ISI ≤ 1.5   — ALL simultaneously, at the FFA layer.

No prior in the 25-prior battery satisfies all four; HOLO-Net meeting them is a
clean existence proof. The minimal-HOLO-Net ablation floor (E041) fails all four
(ISI≈1.0, CSI≈1.19, PWI≈0.78), so any criterion v5 clears is attributable to the
bio-fidelity components.

Run from the repo root (so the manifests' relative image paths resolve):
  cd /workspace/illusionbench-eeg
  python -m holo_net.eval_falsification --checkpoint <ckpt.pt> --tag HOLONET_v5
  # add --minimal if the checkpoint was trained with train.py --minimal
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# paradigm -> (manifest path, pixel baseline, index name)
# Pixel baselines: Thatcher is exactly rotation-invariant ⇒ 1.000; composite
# 1.099 and part-whole 1.202 were measured on the FFHQ pixel model (E023, E024);
# random-bbox is the Thatcher algorithm at non-feature locations ⇒ ≈ 1.000.
PARADIGMS = {
    "thatcher":   ("data/stimuli_ffhq/thatcher_manifest.csv",            1.000, "ISI"),
    "composite":  ("data/stimuli_ffhq_composite/thatcher_manifest.csv",  1.099, "CSI"),
    "partwhole":  ("data/stimuli_ffhq_partwhole/thatcher_manifest.csv",  1.202, "PWI"),
    "randombbox": ("data/stimuli_ffhq_randombbox/thatcher_manifest.csv", 1.000, "ISIrbox"),
}
# FFA-layer falsification criteria (design doc §6), on pixel-corrected values.
CRITERIA = {"thatcher": (">=", 3.0), "composite": (">=", 1.5),
            "partwhole": ("<=", 0.5), "randombbox": ("<=", 1.5)}
LAYERS = ["v1", "v2", "v4", "mfp", "afp", "ffa", "atl"]


def sh(cmd, cwd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main(args):
    repo = Path(args.repo_root)
    tag = args.tag
    per_para: dict[str, dict[str, float]] = {}

    for para, (manifest, pixbase, idx) in PARADIGMS.items():
        emb_dir = repo / f"outputs/embeddings/{para}_{tag}"
        table = repo / f"outputs/tables/{tag}_{para}.csv"
        manifest_abs = repo / manifest

        extract_cmd = [sys.executable, "-m", "holo_net.eval_extract",
                       "--checkpoint", args.checkpoint,
                       "--manifest", str(manifest_abs),
                       "--output_dir", str(emb_dir), "--tag", tag,
                       "--batch_size", str(args.batch_size)]
        if args.minimal:
            extract_cmd.append("--minimal")
        print(f"[{para}] extracting per-layer embeddings ...")
        sh(extract_cmd, cwd=repo)

        print(f"[{para}] computing per-layer index ...")
        sh([sys.executable, "-m", "analysis.compute_metrics",
            "--embedding_dir", str(emb_dir),
            "--manifest", str(manifest_abs), "--output", str(table)], cwd=repo)

        df = pd.read_csv(table)
        vals = {}
        for _, row in df.iterrows():
            layer = str(row["model_id"]).rsplit("_", 1)[-1]
            vals[layer] = float(row["isi_pert"]) / pixbase  # pixel-corrected
        per_para[para] = vals

    # ---- assemble layer × paradigm table (pixel-corrected) ----
    print(f"\n=== HOLO-Net falsification table — {tag} (pixel-corrected) ===")
    print("layer     " + "".join(f"{PARADIGMS[p][2]:>11}" for p in PARADIGMS))
    for layer in LAYERS:
        line = f"{layer:9s}"
        for p in PARADIGMS:
            v = per_para[p].get(layer, float("nan"))
            line += f"{v:11.3f}"
        print(line)
    print("pixel baselines: " + ", ".join(
        f"{p}/{PARADIGMS[p][1]}" for p in PARADIGMS))

    # ---- FFA-layer verdict ----
    print("\n=== FFA-layer falsification verdict (design doc §6) ===")
    all_pass = True
    for p, (op, thr) in CRITERIA.items():
        v = per_para[p].get("ffa", float("nan"))
        ok = (v >= thr) if op == ">=" else (v <= thr)
        all_pass = all_pass and ok
        print(f"  {PARADIGMS[p][2]:8s} = {v:7.3f}   need {op} {thr:<4}  "
              f"{'PASS' if ok else 'FAIL'}")
    verdict = ("*** PASS — HOLO-Net meets the pre-registered falsification ***"
               if all_pass else "FAIL — at least one criterion not met")
    print(f"\n  ALL FOUR SIMULTANEOUSLY: {verdict}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True, help="e.g. HOLONET_v5_step30000")
    p.add_argument("--minimal", action="store_true",
                   help="set if the checkpoint was trained with train.py --minimal")
    p.add_argument("--repo_root", default="/workspace/illusionbench-eeg")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
