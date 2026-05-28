"""Pack and upload Part-Whole v1.4 (kNN 24k, NORMAL_CLONE + per-feature donor match) to HF.

v1.4 changes vs v1.3:
  - Algorithm: cv2.NORMAL_CLONE (was MIXED_CLONE; v1.3 made substitution invisible)
  - Color: L-only ring Reinhard (was full LAB; preserves donor chroma identity)
  - Donor selection: per-feature kNN + InsightFace gender hard match + age ±15
    + per-feature size matching + "skip if both LARGE on that feature"
  - QC harness: ~33-80% pass rate vs v1.3's ~0%

Run on server:
  cd /workspace/illusionbench-eeg
  .venv/bin/python stimuli/classical_cv/upload_partwhole_v1.4.py
"""
from __future__ import annotations
import csv
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path

with open("/workspace/.secrets/hf.env") as f:
    for line in f:
        m = re.match(r"export\s+(\w+)=\"?([^\"]+)\"?", line.strip())
        if m:
            os.environ[m.group(1)] = m.group(2)
# Enable hf_transfer for fast parallel-chunk uploads of big shards.
# v1.3 disabled this; v1.4 estimated ~560 GB, so enable it. If hf_transfer
# is missing the huggingface_hub will warn but fall back to slow uploads.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import upload_file

SRC = Path("/workspace/illusionbench-eeg/data/template_partwhole_v4")
OUT = Path("/workspace/illusionbench-eeg/data/HoloFaceIllusion-Bench-EEG-v1.4")
REPO_ID = "Enhui-1/HoloFaceIllusion-Bench-EEG"
SHARD_SIZE = 50
LOG = Path("/workspace/illusionbench-eeg/runs/upload_v1.4.log")


def log(msg: str):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def is_new_format(tar_path: Path) -> bool:
    try:
        with tarfile.open(tar_path, "r") as tf:
            first = tf.next()
            return first is not None and "/" not in first.name
    except Exception:
        return False


def pack_shard(shard_templates: list[Path], tar_path: Path) -> list[dict]:
    rows = []
    with tarfile.open(tar_path, "w") as tf:
        for td in shard_templates:
            tname = td.name.replace("template_", "")
            mpath = td / "manifest.json"
            if not mpath.exists():
                continue
            by_key: dict[tuple[int, str], dict] = {}
            for row in json.loads(mpath.read_text()):
                k = (int(row["target_idx"]), row["feature"])
                if k not in by_key:
                    by_key[k] = {k2: row[k2] for k2 in
                                  ("template", "target_idx", "feature",
                                   "target_donor", "foil_donor", "ssim_v1v2")}
            for (ti, feat), info in by_key.items():
                sample_key = f"{tname}__t{ti:02d}_{feat}"
                for vk in ["V1", "V2", "V3", "V4"]:
                    fpath = td / f"target{ti:02d}_{feat}_{vk}.png"
                    if fpath.exists():
                        tf.add(str(fpath), arcname=f"{sample_key}.{vk.lower()}.png")
                meta_bytes = json.dumps(info, indent=2).encode("utf-8")
                tinfo = tarfile.TarInfo(name=f"{sample_key}.json")
                tinfo.size = len(meta_bytes)
                tf.addfile(tinfo, io.BytesIO(meta_bytes))
                rows.append({**info, "tar": tar_path.name, "sample_key": sample_key})
    return rows


def collect_rows(shard_templates: list[Path], tar_name: str) -> list[dict]:
    rows = []
    for td in shard_templates:
        tname = td.name.replace("template_", "")
        mpath = td / "manifest.json"
        if not mpath.exists():
            continue
        seen: set[tuple] = set()
        for row in json.loads(mpath.read_text()):
            k = (int(row["target_idx"]), row["feature"])
            if k in seen:
                continue
            seen.add(k)
            sample_key = f"{tname}__t{int(row['target_idx']):02d}_{row['feature']}"
            rows.append({
                "template": row["template"], "target_idx": row["target_idx"],
                "feature": row["feature"], "target_donor": row["target_donor"],
                "foil_donor": row["foil_donor"], "ssim_v1v2": row["ssim_v1v2"],
                "tar": tar_name, "sample_key": sample_key,
            })
    return rows


def upload_with_retry(local_path: Path, repo_path: str, max_tries: int = 5):
    for attempt in range(1, max_tries + 1):
        try:
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=os.environ["HF_TOKEN"],
            )
            return
        except Exception as e:
            log(f"  attempt {attempt} failed: {e}")
            if attempt < max_tries:
                time.sleep(30 * attempt)
            else:
                raise


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    template_dirs = sorted(SRC.glob("template_*"))
    log(f"[init] {len(template_dirs)} template dirs in {SRC}")
    shards = [template_dirs[i:i + SHARD_SIZE]
               for i in range(0, len(template_dirs), SHARD_SIZE)]
    log(f"[init] {len(shards)} shards of {SHARD_SIZE}")

    all_rows = []
    for s_idx, shard in enumerate(shards):
        tar_name = f"partwhole_{s_idx:03d}.tar"
        tar_path = OUT / tar_name
        repo_path = f"v1.4/{tar_name}"

        if tar_path.exists() and is_new_format(tar_path):
            log(f"[ok fmt] {tar_name} already packed — collecting rows only")
            rows = collect_rows(shard, tar_name)
        else:
            if tar_path.exists():
                log(f"[old fmt] {tar_name} — deleting and re-packing")
                tar_path.unlink()
            else:
                log(f"[missing] {tar_name} — packing")
            rows = pack_shard(shard, tar_path)
            sz_gb = tar_path.stat().st_size / 1e9
            log(f"  packed {tar_name} {sz_gb:.2f} GB, {len(rows)} samples")
        all_rows.extend(rows)

        log(f"  uploading {tar_name}...")
        upload_with_retry(tar_path, repo_path)
        log(f"  uploaded {tar_name}")

    if all_rows:
        mpath = OUT / "manifest.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        log(f"manifest: {mpath} ({len(all_rows)} rows)")
        log("uploading manifest.csv...")
        upload_with_retry(mpath, "v1.4/manifest.csv")

    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()
