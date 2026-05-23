"""holo_net/eval_falsification_dinov2_partaware.py — full 4-paradigm §6
falsification on the HOLO-Net v2.2 part-aware FTPC checkpoint."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PARADIGMS = {
    "thatcher":   ("data/stimuli_ffhq/thatcher_manifest.csv",            1.000, "ISI"),
    "composite":  ("data/stimuli_ffhq_composite/thatcher_manifest.csv",  1.099, "CSI"),
    "partwhole":  ("data/stimuli_ffhq_partwhole/thatcher_manifest.csv",  1.202, "PWI"),
    "randombbox": ("data/stimuli_ffhq_randombbox/thatcher_manifest.csv", 1.000, "ISIrbox"),
}
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

        extract_cmd = [sys.executable, "-m", "holo_net.eval_extract_dinov2_partaware",
                       "--checkpoint", args.checkpoint,
                       "--manifest", str(manifest_abs),
                       "--output_dir", str(emb_dir), "--tag", tag,
                       "--batch_size", str(args.batch_size)]
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
            vals[layer] = float(row["isi_pert"]) / pixbase
        per_para[para] = vals

    print(f"\n=== HOLO-Net-DINOv2-PartAware falsification table — {tag} (pixel-corrected) ===")
    print("layer     " + "".join(f"{PARADIGMS[p][2]:>11}" for p in PARADIGMS))
    for layer in LAYERS:
        line = f"{layer:9s}"
        for p in PARADIGMS:
            v = per_para[p].get(layer, float("nan"))
            line += f"{v:11.3f}"
        print(line)
    print("pixel baselines: " + ", ".join(
        f"{p}/{PARADIGMS[p][1]}" for p in PARADIGMS))

    print("\n=== FFA-layer (= δ_concat 1536-d, part-aware FTPC residual) verdict ===")
    all_pass = True
    for p, (op, thr) in CRITERIA.items():
        v = per_para[p].get("ffa", float("nan"))
        ok = (v >= thr) if op == ">=" else (v <= thr)
        all_pass = all_pass and ok
        print(f"  {PARADIGMS[p][2]:8s} = {v:7.3f}   need {op} {thr:<4}  "
              f"{'PASS' if ok else 'FAIL'}")
    verdict = ("*** PASS — HOLO-Net-DINOv2-PartAware meets pre-registered §6 ***"
               if all_pass else "FAIL — at least one criterion not met")
    print(f"\n  ALL FOUR SIMULTANEOUSLY: {verdict}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--repo_root", default="/workspace/illusionbench-eeg")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
