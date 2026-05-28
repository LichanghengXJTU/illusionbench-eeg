"""Re-package v1.2 tars with flat WebDataset naming and upload to HF.

Flat naming convention (WebDataset-compliant):
  {template}__t{idx:02d}_{feature}.{v1|v2|v3|v4}.png
  {template}__t{idx:02d}_{feature}.json

Run on server:
  cd /workspace/illusionbench-eeg
  .venv/bin/python stimuli/classical_cv/repackage_and_upload_v1.2.py
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
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

from huggingface_hub import HfApi, upload_file

SRC = Path("/workspace/illusionbench-eeg/data/template_partwhole_v1")
OUT = Path("/workspace/illusionbench-eeg/data/HoloFaceIllusion-Bench-EEG-v1.2")
REPO_ID = "Enhui-1/HoloFaceIllusion-Bench-EEG"
SHARD_SIZE = 50
LOG = Path("/workspace/illusionbench-eeg/runs/repackage_upload.log")
LOG.parent.mkdir(parents=True, exist_ok=True)

_hf_sizes: dict[str, int] = {}

def _get_hf_sizes():
    global _hf_sizes
    if _hf_sizes:
        return _hf_sizes
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
        _hf_sizes = {s.rfilename: s.size for s in info.siblings}
    except Exception as e:
        log(f"  warn: could not fetch HF sizes: {e}")
        _hf_sizes = {}
    return _hf_sizes

def _hf_has(repo_path: str, local_path: Path) -> bool:
    sizes = _get_hf_sizes()
    hf_sz = sizes.get(repo_path, 0)
    return hf_sz > 0 and abs(hf_sz - local_path.stat().st_size) <= 1024


def log(msg: str):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def pack_shard(shard_templates: list[Path], tar_path: Path) -> list[dict]:
    rows = []
    with tarfile.open(tar_path, "w") as tf:
        for td in shard_templates:
            tname = td.name.replace("template_", "")
            mpath = td / "manifest.json"
            if not mpath.exists():
                continue
            meta_list = json.loads(mpath.read_text())
            by_key: dict[tuple[int, str], dict] = {}
            for row in meta_list:
                k = (int(row["target_idx"]), row["feature"])
                if k not in by_key:
                    by_key[k] = {
                        "template": row["template"],
                        "target_idx": row["target_idx"],
                        "feature": row["feature"],
                        "target_donor": row["target_donor"],
                        "foil_donor": row["foil_donor"],
                        "ssim_v1v2": row["ssim_v1v2"],
                    }
            for (ti, feat), info in by_key.items():
                sample_key = f"{tname}__t{ti:02d}_{feat}"
                for vk in ["V1", "V2", "V3", "V4"]:
                    fpath = td / f"target{ti:02d}_{feat}_{vk}.png"
                    if not fpath.exists():
                        continue
                    tf.add(str(fpath), arcname=f"{sample_key}.{vk.lower()}.png")
                meta_bytes = json.dumps(info, indent=2).encode("utf-8")
                tinfo = tarfile.TarInfo(name=f"{sample_key}.json")
                tinfo.size = len(meta_bytes)
                tf.addfile(tinfo, io.BytesIO(meta_bytes))
                rows.append({**info, "tar": tar_path.name, "sample_key": sample_key})
    return rows


def _collect_rows_from_source(shard_templates: list[Path], tar_name: str) -> list[dict]:
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
    log(f"[init] {len(template_dirs)} template dirs")
    shards = [template_dirs[i:i + SHARD_SIZE]
               for i in range(0, len(template_dirs), SHARD_SIZE)]
    log(f"[init] {len(shards)} shards of {SHARD_SIZE}")

    all_rows = []
    for s_idx, shard in enumerate(shards):
        tar_name = f"partwhole_{s_idx:02d}.tar"
        tar_path = OUT / tar_name
        repo_path = f"v1.2/{tar_name}"

        # Skip pack if local tar already exists (was packed in previous run)
        if tar_path.exists():
            log(f"[skip pack] {tar_name} already local ({tar_path.stat().st_size/1e9:.2f} GB)")
            # Still need to collect rows for manifest
            rows = _collect_rows_from_source(shard, tar_name)
            all_rows.extend(rows)
        else:
            log(f"packing {tar_name} ({len(shard)} templates)...")
            rows = pack_shard(shard, tar_path)
            all_rows.extend(rows)
            sz_gb = tar_path.stat().st_size / 1e9
            log(f"  wrote {tar_name} {sz_gb:.2f} GB, {len(rows)} samples")

        # Skip upload if HF already has it with matching size
        if _hf_has(repo_path, tar_path):
            log(f"[skip upload] {tar_name} already on HF")
            continue

        log(f"  uploading {tar_name}...")
        upload_with_retry(tar_path, repo_path)
        log(f"  uploaded {tar_name}")
        _hf_sizes.pop(repo_path, None)  # invalidate cache entry

    # Write manifest
    if all_rows:
        mpath = OUT / "manifest.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        log(f"manifest: {mpath} ({len(all_rows)} rows)")
        log("uploading manifest.csv...")
        upload_with_retry(mpath, "v1.2/manifest.csv")

    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()
