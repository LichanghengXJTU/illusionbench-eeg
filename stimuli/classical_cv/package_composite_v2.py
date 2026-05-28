"""stimuli/classical_cv/package_composite_v2.py — pack the flat dump of
composite V1-V4 PNG + JSON files into webdataset tars.

Input layout (from composite_v2_parallel.py Phase B):
  data/composite_v2_pngs/
    NNNNN__dMMMMM.v1.png
    NNNNN__dMMMMM.v2.png
    NNNNN__dMMMMM.v3.png
    NNNNN__dMMMMM.v4.png
    NNNNN__dMMMMM.json
    NNNNN__dMMMMM.contact.png      (← excluded from tar)
    ...

Output layout (webdataset):
  data/composite_v2/
    v2.0/composite_000.tar
    v2.0/composite_001.tar
    ...
    v2.0/manifest.csv

Each tar shard contains ~500 (template, donor) pairs = ~2500 files
(5 per pair: v1, v2, v3, v4, json). Average shard size ≈ 280 MB.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import tarfile
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/composite_v2_pngs")
    ap.add_argument("--dst", default="data/composite_v2")
    ap.add_argument("--pairs_per_shard", type=int, default=500)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) / "v2.0"
    dst.mkdir(parents=True, exist_ok=True)

    json_files = sorted(src.glob("*.json"))
    print(f"[pack] found {len(json_files)} (T, D) pairs in {src}")
    rows = []
    shard_idx = 0
    written = 0
    cur_tar = None
    cur_tar_path = None

    def _new_tar():
        nonlocal cur_tar, cur_tar_path, shard_idx
        if cur_tar is not None:
            cur_tar.close()
            print(f"  closed {cur_tar_path}  "
                  f"({cur_tar_path.stat().st_size/1e6:.1f} MB)")
            shard_idx += 1
        cur_tar_path = dst / f"composite_{shard_idx:03d}.tar"
        cur_tar = tarfile.open(cur_tar_path, "w")

    _new_tar()
    t0 = time.monotonic()
    for i, jp in enumerate(json_files):
        base = jp.stem  # NNNNN__dMMMMM
        # Add v1-v4 PNGs + JSON to current tar
        for ext in ("v1.png", "v2.png", "v3.png", "v4.png", "json"):
            fpath = src / f"{base}.{ext}"
            if not fpath.exists():
                print(f"  WARN missing {fpath}")
                continue
            cur_tar.add(fpath, arcname=fpath.name)
            written += 1
        # Manifest row
        try:
            meta = json.loads(jp.read_text())
            rows.append({
                "template": meta["template"], "donor": meta["donor"],
                "sample_key": base, "tar": cur_tar_path.name,
                "y_cut": meta["y_cut"], "face_w": meta["face_w"],
                "mis_dx_px": meta["mis_dx_px"],
                "ssim_v1v2": meta.get("ssim_v1v2", -1),
                "seam_disc": meta.get("seam_disc", -1),
                "lab_delta_cut": meta.get("lab_delta_cut", -1),
                "template_yaw_deg": meta.get("template_yaw_deg", float("nan")),
                "donor_yaw_deg": meta.get("donor_yaw_deg", float("nan")),
            })
        except Exception as e:
            print(f"  WARN bad json {jp}: {e}")
        if (i + 1) % args.pairs_per_shard == 0:
            _new_tar()
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(json_files)}] written={written} files  "
                  f"elapsed={(time.monotonic()-t0):.0f}s", flush=True)

    if cur_tar is not None:
        cur_tar.close()
        print(f"  closed {cur_tar_path}  "
              f"({cur_tar_path.stat().st_size/1e6:.1f} MB)")

    # Manifest
    mpath = dst / "manifest.csv"
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[pack] DONE")
    print(f"  shards: {shard_idx + 1}")
    print(f"  pairs: {len(json_files)}  files: {written}")
    print(f"  manifest: {mpath}  ({len(rows)} rows)")
    print(f"  elapsed: {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
