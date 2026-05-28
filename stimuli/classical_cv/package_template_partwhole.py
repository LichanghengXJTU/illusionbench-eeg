"""stimuli/classical_cv/package_template_partwhole.py — pack
template-based PW v1.2 into webdataset tar shards for HF release.

Input layout (from template_partwhole_parallel.py):
  data/template_partwhole_v1/
    template_<NAME>/
      target00_eye_V1.png  ...  target05_mouth_V4.png   (6 × 3 × 4 = 72)
      manifest.json
    ...

Output layout (HF v1.2):
  partwhole_NN.tar          (one tar per ~50 templates)
  Per-sample (webdataset convention): one sample = one (template, target, feature)
    sample key  = `{template_NAME}/t{target_idx:02d}_{feature}`
    sample files:
      V1.png  V2.png  V3.png  V4.png  meta.json
"""
from __future__ import annotations
import argparse
import csv
import json
import tarfile
from pathlib import Path

import numpy as np


def main(args):
    src = Path(args.input_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    template_dirs = sorted(src.glob("template_*"))
    print(f"[init] {len(template_dirs)} template dirs in {src}")
    # Group templates into tar shards
    shard_size = args.shard_size
    shards: list[list[Path]] = [
        template_dirs[i:i + shard_size]
        for i in range(0, len(template_dirs), shard_size)
    ]
    print(f"[init] {len(shards)} tar shards (size {shard_size})")
    manifest_rows = []
    for s_idx, shard in enumerate(shards):
        tar_name = f"partwhole_{s_idx:02d}.tar"
        tar_path = out / tar_name
        if tar_path.exists() and not args.overwrite:
            print(f"  [skip] {tar_name} exists")
            for td in shard:
                tname = td.name.replace("template_", "")
                _emit_manifest_rows(td, tname, tar_name, manifest_rows)
            continue
        print(f"  packing {tar_name} ({len(shard)} templates)...")
        with tarfile.open(tar_path, "w") as tf:
            for td in shard:
                tname = td.name.replace("template_", "")
                meta = json.loads((td / "manifest.json").read_text()) \
                       if (td / "manifest.json").exists() else []
                # Group manifest entries by (target_idx, feature)
                by_key: dict[tuple[int, str], dict] = {}
                for row in meta:
                    k = (int(row["target_idx"]), row["feature"])
                    if k not in by_key:
                        by_key[k] = {
                            "template":      row["template"],
                            "target_idx":    row["target_idx"],
                            "feature":       row["feature"],
                            "target_donor":  row["target_donor"],
                            "foil_donor":    row["foil_donor"],
                            "ssim_v1v2":     row["ssim_v1v2"],
                        }
                for (ti, feat), info in by_key.items():
                    sample_key = f"{tname}__t{ti:02d}_{feat}"
                    for vk in ["V1", "V2", "V3", "V4"]:
                        fpath = td / f"target{ti:02d}_{feat}_{vk}.png"
                        if not fpath.exists():
                            continue
                        tf.add(str(fpath),
                               arcname=f"{sample_key}.{vk.lower()}.png")
                    # Sample metadata
                    meta_str = json.dumps(info, indent=2)
                    meta_bytes = meta_str.encode("utf-8")
                    tinfo = tarfile.TarInfo(name=f"{sample_key}.json")
                    tinfo.size = len(meta_bytes)
                    import io
                    tf.addfile(tinfo, io.BytesIO(meta_bytes))
                    manifest_rows.append({
                        **info, "tar": tar_name, "sample_key": sample_key,
                    })
        sz_mb = tar_path.stat().st_size / 1e6
        print(f"    wrote {tar_path.name}  ({sz_mb:.1f} MB)")
    # Global manifest
    if manifest_rows:
        mpath = out / "manifest.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader(); w.writerows(manifest_rows)
        print(f"[done] {len(manifest_rows)} samples across {len(shards)} shards")
        print(f"  manifest: {mpath}")
        print(f"  total size: "
              f"{sum(p.stat().st_size for p in out.glob('partwhole_*.tar'))/1e9:.2f} GB")


def _emit_manifest_rows(td: Path, tname: str, tar_name: str,
                         manifest_rows: list):
    """Add manifest entries from existing template dir (for skipped shards)."""
    mp = td / "manifest.json"
    if not mp.exists():
        return
    meta = json.loads(mp.read_text())
    seen = set()
    for row in meta:
        k = (row["target_idx"], row["feature"])
        if k in seen:
            continue
        seen.add(k)
        sample_key = f"{tname}__t{int(row['target_idx']):02d}_{row['feature']}"
        manifest_rows.append({
            "template":      row["template"],
            "target_idx":    row["target_idx"],
            "feature":       row["feature"],
            "target_donor":  row["target_donor"],
            "foil_donor":    row["foil_donor"],
            "ssim_v1v2":     row["ssim_v1v2"],
            "tar":           tar_name,
            "sample_key":    sample_key,
        })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",
                   default="data/template_partwhole_v1")
    p.add_argument("--output_dir",
                   default="data/HoloFaceIllusion-Bench-EEG-v1.2")
    p.add_argument("--shard_size", type=int, default=50,
                   help="number of templates per tar shard")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
