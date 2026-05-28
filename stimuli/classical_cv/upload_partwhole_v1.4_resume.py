"""Resume v1.4 upload using upload_large_folder (avoids per-shard commit limit).

Background: upload_partwhole_v1.4.py hit HF's 128-commits-per-hour rate limit
after 127 shards (each shard = 1 commit). upload_large_folder batches multiple
files into a single commit, so for ~342 remaining shards we get ~14 commits.

Workflow:
  1. Determine which shards are NOT yet on HF (skip 0..126; we know they're up).
  2. Pack missing shards (template dir → tar with WebDataset layout).
  3. upload_large_folder on the dir — handles parallel uploads + commit batching.
  4. Build complete manifest.csv from per-template manifest.json files (all 469).
  5. Upload manifest.csv.
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
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, upload_file

SRC = Path("/workspace/illusionbench-eeg/data/template_partwhole_v4")
OUT = Path("/workspace/illusionbench-eeg/data/HoloFaceIllusion-Bench-EEG-v1.4")
REPO_ID = "Enhui-1/HoloFaceIllusion-Bench-EEG"
SHARD_SIZE = 50
LOG = Path("/workspace/illusionbench-eeg/runs/upload_v1.4_resume.log")

# Shards already uploaded — based on first run's log
ALREADY_UPLOADED = set(range(0, 127))  # partwhole_000.tar .. partwhole_126.tar


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
    """Collect manifest rows for shards we don't need to re-pack (already uploaded)."""
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    template_dirs = sorted(SRC.glob("template_*"))
    log(f"[init] {len(template_dirs)} template dirs in {SRC}")
    shards = [template_dirs[i:i + SHARD_SIZE]
               for i in range(0, len(template_dirs), SHARD_SIZE)]
    log(f"[init] {len(shards)} shards of {SHARD_SIZE}")
    log(f"[init] {len(ALREADY_UPLOADED)} shards already on HF, "
        f"{len(shards) - len(ALREADY_UPLOADED)} to pack+upload")

    all_rows: list[dict] = []
    to_upload: list[Path] = []

    for s_idx, shard in enumerate(shards):
        tar_name = f"partwhole_{s_idx:03d}.tar"
        tar_path = OUT / tar_name

        if s_idx in ALREADY_UPLOADED:
            rows = collect_rows(shard, tar_name)
            all_rows.extend(rows)
            continue

        if not tar_path.exists():
            log(f"[pack] {tar_name}")
            rows = pack_shard(shard, tar_path)
            sz_gb = tar_path.stat().st_size / 1e9
            log(f"  packed {tar_name} {sz_gb:.2f} GB, {len(rows)} samples")
        else:
            log(f"[skip-pack] {tar_name} exists")
            rows = collect_rows(shard, tar_name)
        all_rows.extend(rows)
        to_upload.append(tar_path)

    log(f"[pack-done] {len(to_upload)} tars to upload, "
        f"total size: {sum(p.stat().st_size for p in to_upload)/1e9:.1f} GB")

    # HF rate-limit cooldown: previous run hit 429 (128 commits/hour).
    # Wait until the 1-hour rolling window has cleared. Read first commit time
    # from prior log if present.
    cooldown_until = float(os.environ.get("HF_RESUME_COOLDOWN_UNTIL", "0"))
    if cooldown_until > 0:
        wait = cooldown_until - time.time()
        if wait > 0:
            log(f"[rate-limit] sleeping {wait/60:.1f} min until "
                f"{time.strftime('%H:%M:%S', time.localtime(cooldown_until))} "
                f"to clear HF commit rate limit (128/h)")
            time.sleep(wait)
        log(f"[rate-limit] cooldown complete, proceeding")

    # Upload all remaining shards in one batched upload (handles commit batching)
    api = HfApi(token=os.environ["HF_TOKEN"])
    log(f"[upload] starting upload_large_folder on {OUT}")
    log(f"         repo_id={REPO_ID}  path_in_repo='v1.4'")
    api.upload_large_folder(
        folder_path=str(OUT),
        repo_id=REPO_ID,
        repo_type="dataset",
        # Only upload partwhole_*.tar files (skip any leftover artifacts)
        allow_patterns=["partwhole_*.tar"],
        ignore_patterns=[f"partwhole_{i:03d}.tar" for i in ALREADY_UPLOADED],
    )
    log(f"[upload] done")

    # Manifest with ALL rows (all 469 shards)
    mpath = OUT / "manifest.csv"
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    log(f"manifest: {mpath} ({len(all_rows)} rows)")
    log("uploading manifest.csv...")
    # Single file upload (one commit, fine)
    upload_file(
        path_or_fileobj=str(mpath),
        path_in_repo="v1.4/manifest.csv",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()
