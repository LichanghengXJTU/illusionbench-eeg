"""stimuli/classical_cv/upload_composite_v2.py — composite v2.0 upload with
hf-CLI subprocess + os.read non-blocking stdout tracking + tqdm bytes
extraction watchdog.

Per the user request: if no upload byte growth in stall_seconds (~30-60s),
kill the upload and restart. Restart is cheap because LFS uploads resume
from completed chunks.

Implementation:
  - Run `hf upload` as subprocess.
  - Background thread: read subprocess stdout non-blocking via os.read.
    For each chunk: append to a buffer, parse tqdm progress patterns to
    extract current bytes uploaded.
  - Main loop: every check_interval, if current bytes hasn't grown for
    stall_seconds, terminate subprocess and restart.
"""
from __future__ import annotations
import argparse
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi


# Patterns to extract progress from hf upload stdout
# Examples:
#   "  ...v2/v2.0/composite_004.tar:   8%|▊         | 70.5MB /  849MB"
#   "New Data Upload               :  35%|███▌      | 70.5MB /  201MB,  106kB/s"
_RE_BYTES = re.compile(
    r"(\d+(?:\.\d+)?)\s*(B|KB|kB|MB|GB)\s*/\s*(\d+(?:\.\d+)?)\s*(B|KB|kB|MB|GB)"
)
_MULT = {"B": 1, "KB": 1024, "kB": 1024, "MB": 1024**2, "GB": 1024**3}


def parse_progress_bytes(s: str) -> tuple[int, int] | None:
    """Extract (current_bytes, total_bytes) from a tqdm progress string."""
    m = _RE_BYTES.search(s)
    if not m:
        return None
    cur = float(m.group(1)) * _MULT.get(m.group(2), 1)
    tot = float(m.group(3)) * _MULT.get(m.group(4), 1)
    return int(cur), int(tot)


def _repo_files(api, repo, subdir):
    try:
        info = api.list_repo_tree(repo, path_in_repo=subdir,
                                   repo_type="dataset", recursive=False,
                                   expand=True)
        out = {}
        for e in info:
            if e.path.startswith(subdir + "/") and not e.path.endswith("/"):
                out[e.path.split("/")[-1]] = int(getattr(e, "size", 0) or 0)
        return out
    except Exception:
        return {}


def upload_with_watchdog(src: Path, repo: str, subdir: str, token: str,
                         stall_seconds: float, check_interval: float,
                         max_attempts: int) -> bool:
    """Single attempt: launch hf upload + watchdog. Returns True if all files
    completed in this attempt. Note: hf upload is idempotent (skips
    already-uploaded files), so we just keep launching until all files exist
    on HF."""
    api = HfApi(token=token)
    local_files = sorted([p.name for p in src.iterdir() if p.is_file()])
    local_sizes = {p.name: p.stat().st_size for p in src.iterdir() if p.is_file()}

    for attempt in range(max_attempts):
        # Check what's already done
        sizes = _repo_files(api, repo, subdir)
        missing = [n for n in local_files if sizes.get(n, -1) != local_sizes[n]]
        if not missing:
            print(f"[upload] all {len(local_files)} files complete on HF", flush=True)
            return True
        print(f"[upload] [attempt {attempt+1}/{max_attempts}] missing {len(missing)} / "
              f"{len(local_files)} → launching hf upload", flush=True)

        env = os.environ.copy()
        env["HF_TOKEN"] = token
        cmd = ["hf", "upload", repo, str(src), subdir,
               "--repo-type", "dataset",
               "--commit-message", f"composite v2.0 batch attempt {attempt+1}"]
        proc = subprocess.Popen(cmd, env=env,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 bufsize=0)
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)

        buf = b""
        last_bytes = -1
        last_progress_t = time.monotonic()
        t_start = time.monotonic()
        last_print_t = 0.0

        try:
            while proc.poll() is None:
                # Read non-blocking, up to 64KB
                try:
                    r, _, _ = select.select([fd], [], [], check_interval)
                    if r:
                        chunk = os.read(fd, 65536)
                        if chunk:
                            buf += chunk
                    else:
                        chunk = b""
                except (OSError, ValueError):
                    chunk = b""

                # Parse buf for tqdm progress
                # Split on \r and \n to catch tqdm updates
                lines = buf.replace(b"\r", b"\n").split(b"\n")
                buf = lines[-1]  # incomplete trailing line
                for ln in lines[:-1]:
                    s = ln.decode("utf-8", errors="ignore").strip()
                    if not s:
                        continue
                    pr = parse_progress_bytes(s)
                    if pr is not None:
                        cur, tot = pr
                        if cur > last_bytes:
                            last_bytes = cur
                            last_progress_t = time.monotonic()
                            now = time.monotonic()
                            if now - last_print_t > 6.0:
                                print(f"  [+] {cur/1e6:.0f}MB / {tot/1e6:.0f}MB",
                                      flush=True)
                                last_print_t = now
                    elif any(k in s for k in ("Start hashing", "Finished hashing",
                                                "Processing", "Uploading",
                                                "Commit", "Pushed", "LFS")):
                        # Any hf output line counts as activity
                        last_progress_t = time.monotonic()
                        print(f"  hf> {s}", flush=True)

                # Stall check
                stall_for = time.monotonic() - last_progress_t
                if stall_for > stall_seconds:
                    el = time.monotonic() - t_start
                    print(f"  [STALL] no byte growth in {stall_for:.0f}s "
                          f"(attempt elapsed {el:.0f}s) → kill", flush=True)
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

            # Drain remaining stdout
            try:
                rest = b""
                while True:
                    try:
                        c = os.read(fd, 65536)
                    except (OSError, BlockingIOError):
                        break
                    if not c:
                        break
                    rest += c
                if rest:
                    print(f"  hf> (final) {rest[:500].decode('utf-8', errors='ignore').strip()}",
                          flush=True)
            except Exception:
                pass

            rc = proc.returncode
            el = time.monotonic() - t_start
            print(f"[upload] attempt {attempt+1} exited rc={rc} in {el:.0f}s",
                  flush=True)
        except KeyboardInterrupt:
            proc.kill()
            raise
        time.sleep(3)

    sizes = _repo_files(api, repo, subdir)
    missing = [n for n in local_files if sizes.get(n, -1) != local_sizes[n]]
    if not missing:
        return True
    print(f"[upload] FAILED — still missing: {missing}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo_subdir", default="v2.0")
    ap.add_argument("--stall_seconds", type=float, default=60.0)
    ap.add_argument("--check_interval", type=float, default=5.0)
    ap.add_argument("--max_attempts", type=int, default=50)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        tf = Path("/tmp/hf_env.sh")
        if tf.exists():
            for line in tf.read_text().splitlines():
                if "HF_TOKEN" in line and "=" in line:
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        print("[upload] no HF_TOKEN", flush=True); sys.exit(1)

    src = Path(args.src)
    files = [p for p in src.iterdir() if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"[upload] {len(files)} files ({total/1e9:.2f}GB) → "
          f"{args.repo}:{args.repo_subdir}/  stall={args.stall_seconds}s",
          flush=True)

    ok = upload_with_watchdog(src, args.repo, args.repo_subdir, token,
                                args.stall_seconds, args.check_interval,
                                args.max_attempts)
    if not ok:
        sys.exit(1)
    print(f"[upload] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
