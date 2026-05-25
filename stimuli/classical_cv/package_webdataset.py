"""stimuli/classical_cv/package_webdataset.py — pack 26k IDs into FFHQ-style tars.

Output convention matches `gaunernst/ffhq-1024-wds` upstream:
  - 70 tars, named by leading FFHQ index: 00000.tar (covers ffhq_00000..00999),
    01000.tar (covers ffhq_01000..01999), ..., 69000.tar
  - Per-sample webdataset entries (one entry = one identity):
       {ffhq_name}/V1_upright_normal.png
       {ffhq_name}/V2_upright_thatched.png
       {ffhq_name}/V3_inverted_normal.png
       {ffhq_name}/V4_inverted_thatched.png
       {ffhq_name}/landmarks.json
  - Plus a top-level `manifest.csv` (NOT inside any tar) listing all included
    ffhq_names + source-tar mapping (for downstream re-use).

Idempotent — skips tars already present in output dir.
"""
from __future__ import annotations
import argparse
import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path


def main(args):
    src_root = Path(args.input_dir)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    id_dirs = sorted(src_root.glob("ffhq_*"))
    print(f"[init] {len(id_dirs)} identity dirs found")

    # Group identities by leading FFHQ index (every 1000 IDs → one tar)
    groups: dict[int, list[Path]] = defaultdict(list)
    for d in id_dirs:
        try:
            ffhq_idx = int(d.name.split("_")[-1])
        except ValueError:
            continue
        tar_key = (ffhq_idx // 1000) * 1000
        groups[tar_key].append(d)

    print(f"[init] grouped into {len(groups)} tars")
    manifest_rows = []
    for tar_key in sorted(groups.keys()):
        tar_name = f"{tar_key:05d}.tar"
        tar_path = out_root / tar_name
        if tar_path.exists() and not args.overwrite:
            print(f"  [skip] {tar_name} exists")
            for d in groups[tar_key]:
                manifest_rows.append({"ffhq_name": d.name.replace("ffhq_", ""),
                                        "tar": tar_name})
            continue
        ids = sorted(groups[tar_key], key=lambda p: p.name)
        print(f"  packing {tar_name} ({len(ids)} identities)...")
        with tarfile.open(tar_path, "w") as tf:
            for d in ids:
                ffhq_name = d.name.replace("ffhq_", "")
                # Required per-sample files
                file_map = {
                    "V1_upright_normal.png":   d / "thatcher_V1.png",
                    "V2_upright_thatched.png": d / "thatcher_V2.png",
                    "V3_inverted_normal.png":  d / "thatcher_V3.png",
                    "V4_inverted_thatched.png":d / "thatcher_V4.png",
                    "landmarks.json":          d / "landmarks.json",
                }
                missing = [k for k, p in file_map.items() if not p.exists()]
                if missing:
                    print(f"    [skip] {ffhq_name}: missing {missing}")
                    continue
                for arc_name, src_path in file_map.items():
                    tf.add(str(src_path), arcname=f"{ffhq_name}/{arc_name}")
                manifest_rows.append({"ffhq_name": ffhq_name, "tar": tar_name})

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader(); w.writerows(manifest_rows)
    print(f"\n[done] {len(manifest_rows)} identities packaged into "
           f"{len(groups)} tars → {out_root}")
    print(f"       manifest: {manifest_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/thatcher_ffhq_70k_parallel")
    p.add_argument("--output_dir",
                   default="data/HoloFaceIllusion-Bench-EEG-v1.0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
