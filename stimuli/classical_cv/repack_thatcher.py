"""Repack Thatcher tars from `NNNNN/X.ext` (subdir-per-identity) layout
into webdataset-compliant `NNNNN.X.ext` (shared-prefix-per-sample) layout.

The original layout causes HF's webdataset loader to treat each PNG as its
own sample (different key per file), which fails the viewer with
'files don't share the same prefix'. The new layout groups V1/V2/V3/V4/json
of the same identity into a single sample with named fields.

Usage:
    python repack_thatcher.py --src /workspace/.../v1.0 --dst /workspace/.../v1.1 --workers 32
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import tarfile
import time
from pathlib import Path


def repack_one(args_tuple):
    src_path, dst_path = args_tuple
    if dst_path.exists() and dst_path.stat().st_size > 1024:
        return (src_path.name, "skip_exists", dst_path.stat().st_size)
    try:
        n_kept = 0
        with tarfile.open(src_path, "r") as src, \
             tarfile.open(dst_path, "w") as dst:
            while True:
                try:
                    m = src.next()
                except (tarfile.ReadError, EOFError, OSError):
                    break
                if m is None:
                    break
                if not m.isfile():
                    continue
                name = m.name
                if "/" not in name:
                    continue
                # Replace ALL slashes with dots so key = first-component
                new_name = name.replace("/", ".")
                # Strip optional ffhq_ prefix if present (clean up)
                if new_name.startswith("ffhq_"):
                    new_name = new_name[5:]
                # Lowercase the V-prefix for cleaner field names
                # e.g. "00002.V1_upright_normal.png" -> "00002.v1_upright_normal.png"
                parts = new_name.split(".", 1)
                if len(parts) == 2:
                    key, rest = parts
                    if rest.startswith(("V1_", "V2_", "V3_", "V4_")):
                        rest = rest[0].lower() + rest[1:]
                    new_name = f"{key}.{rest}"
                try:
                    f = src.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except (tarfile.ReadError, EOFError, OSError):
                    break
                if len(data) != m.size or len(data) == 0:
                    continue
                # Build new member with renamed path
                import io
                new_m = tarfile.TarInfo(new_name)
                new_m.size = len(data)
                new_m.mtime = m.mtime
                new_m.mode = m.mode
                dst.addfile(new_m, io.BytesIO(data))
                n_kept += 1
        return (src_path.name, "ok", n_kept)
    except Exception as e:
        try:
            if dst_path.exists():
                dst_path.unlink()
        except Exception:
            pass
        return (src_path.name, "error", repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    tars = sorted(src.glob("*.tar"))
    print(f"[repack] {len(tars)} tars in {src}")
    tasks = [(t, dst / t.name) for t in tars]

    t0 = time.monotonic()
    done = 0
    total_kept = 0
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(repack_one, tasks):
            name, status, info = r
            done += 1
            if status == "ok":
                total_kept += info
            print(f"  [{done}/{len(tars)}] {name}: {status} ({info})", flush=True)
    print(f"[repack] DONE: {done} tars, {total_kept} files, "
          f"elapsed {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
