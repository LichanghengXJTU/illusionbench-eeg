"""Force re-pack any old-format shards and upload all 20 to HF unconditionally.

Old format: ffhq_13250/t00_eye/V1.png  (slash-separated, breaks HF viewer)
New format: 13250__t00_eye.v1.png      (flat, WebDataset-compliant)
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

from huggingface_hub import upload_file

SRC = Path("/workspace/illusionbench-eeg/data/template_partwhole_v1")
OUT = Path("/workspace/illusionbench-eeg/data/HoloFaceIllusion-Bench-EEG-v1.2")
REPO_ID = "Enhui-1/HoloFaceIllusion-Bench-EEG"
SHARD_SIZE = 50
LOG = Path("/workspace/illusionbench-eeg/runs/force_repack_upload.log")


def log(msg: str):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def is_new_format(tar_path: Path) -> bool:
    """Check if first entry uses flat naming (new) vs slash paths (old)."""
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
    log(f"[init] {len(template_dirs)} template dirs")
    shards = [template_dirs[i:i + SHARD_SIZE]
               for i in range(0, len(template_dirs), SHARD_SIZE)]
    log(f"[init] {len(shards)} shards")

    all_rows = []
    for s_idx, shard in enumerate(shards):
        tar_name = f"partwhole_{s_idx:02d}.tar"
        tar_path = OUT / tar_name
        repo_path = f"v1.2/{tar_name}"

        # Re-pack if missing or old format
        if tar_path.exists() and is_new_format(tar_path):
            log(f"[ok fmt] {tar_name} already new-format — collecting rows only")
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

        # Force-upload unconditionally
        log(f"  uploading {tar_name}...")
        upload_with_retry(tar_path, repo_path)
        log(f"  uploaded {tar_name}")

    # Manifest
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
