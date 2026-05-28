#!/bin/bash
# upload_composite_v2_stall_retry.sh
# Upload v2.0/composite_*.tar to HF with 30s stall detection.
# Kills hf upload if no progress for STALL_SEC, retries up to MAX_RETRIES.
# Progress = total bytes of v2.0/ files visible via HF API (committed bytes
# don't grow during upload; instead we monitor (a) hf process stdout growth
# AND (b) network TX bytes on eth0 — either growing = alive).

set -u

SRC="${SRC:-/workspace/illusionbench-eeg/data/composite_v2/v2.0}"
REPO="${REPO:-Enhui-1/HoloFaceIllusion-Bench-EEG}"
DEST="${DEST:-v2.0}"
STALL_SEC="${STALL_SEC:-30}"
MAX_RETRIES="${MAX_RETRIES:-30}"
LOG_DIR="${LOG_DIR:-/workspace/illusionbench-eeg/logs}"

source /workspace/.secrets/hf.env

export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_FIXED_UPLOAD_CONCURRENCY=64
export HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION=32
export HF_HUB_ENABLE_HF_TRANSFER=0   # use xet
export HF_HUB_DISABLE_PROGRESS_BARS=0

mkdir -p "$LOG_DIR"

# Find the network interface with the default route
IFACE=$(ip route | awk '/^default/ {print $5; exit}')
[ -z "$IFACE" ] && IFACE=eth0
echo "[wrapper] using interface: $IFACE"

# Helper: get cumulative TX bytes for IFACE
tx_bytes() {
  cat /sys/class/net/$IFACE/statistics/tx_bytes 2>/dev/null || echo 0
}

# Helper: count remote v2.0/composite_*.tar files
remote_count() {
  /workspace/illusionbench-eeg/.venv/bin/python - <<'PYEOF' 2>/dev/null
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
try:
    files = api.list_repo_files(repo_id="Enhui-1/HoloFaceIllusion-Bench-EEG", repo_type="dataset")
    print(sum(1 for f in files if f.startswith("v2.0/composite_") and f.endswith(".tar")))
except Exception as e:
    print(-1)
PYEOF
}

attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
  log="$LOG_DIR/upload_attempt_${attempt}.log"
  echo "[wrapper] === ATTEMPT $attempt / $MAX_RETRIES ==="
  echo "[wrapper] log: $log"
  rc_before=$(remote_count)
  echo "[wrapper] remote tars before: $rc_before"

  # Launch hf upload in background
  /workspace/illusionbench-eeg/.venv/bin/hf upload "$REPO" "$SRC" "$DEST" \
    --repo-type dataset \
    --commit-message "composite v2.0 full-scale (attempt $attempt)" \
    > "$log" 2>&1 &
  UP_PID=$!
  echo "[wrapper] hf upload pid=$UP_PID"

  # Monitor: every 5s check (a) log mtime growth and (b) TX bytes growth
  last_log_size=$(stat -c%s "$log" 2>/dev/null || echo 0)
  last_tx=$(tx_bytes)
  stall_acc=0
  ok=0
  while true; do
    sleep 5
    # process still alive?
    if ! kill -0 $UP_PID 2>/dev/null; then
      wait $UP_PID
      rc=$?
      echo "[wrapper] hf upload exited rc=$rc"
      if [ $rc -eq 0 ]; then
        echo "[wrapper] SUCCESS"
        ok=1
      fi
      break
    fi
    cur_log_size=$(stat -c%s "$log" 2>/dev/null || echo 0)
    cur_tx=$(tx_bytes)
    tx_delta=$((cur_tx - last_tx))
    log_delta=$((cur_log_size - last_log_size))
    # Alive if either log grew OR TX grew > 1MB in 5s (xet ramp)
    if [ "$log_delta" -gt 0 ] || [ "$tx_delta" -gt 1048576 ]; then
      stall_acc=0
      echo "[wrapper] alive  log+=$log_delta  tx+=$((tx_delta/1024))KB/5s"
    else
      stall_acc=$((stall_acc + 5))
      echo "[wrapper] no-growth ${stall_acc}s  (log+=$log_delta  tx+=$((tx_delta))B)"
    fi
    last_log_size=$cur_log_size
    last_tx=$cur_tx
    if [ "$stall_acc" -ge "$STALL_SEC" ]; then
      echo "[wrapper] STALL >$STALL_SEC s → killing PID $UP_PID"
      kill -TERM $UP_PID 2>/dev/null
      sleep 3
      kill -KILL $UP_PID 2>/dev/null
      wait $UP_PID 2>/dev/null
      break
    fi
  done

  rc_after=$(remote_count)
  echo "[wrapper] remote tars after: $rc_after"
  if [ "$ok" = "1" ]; then
    # double-check all tars uploaded
    local_count=$(ls "$SRC"/composite_*.tar 2>/dev/null | wc -l)
    if [ "$rc_after" -ge "$local_count" ] && [ "$rc_after" -ge 1 ]; then
      echo "[wrapper] DONE: $rc_after / $local_count tars on HF"
      exit 0
    fi
    echo "[wrapper] partial: only $rc_after/$local_count on HF, retrying"
  fi
  attempt=$((attempt + 1))
  sleep 2
done

echo "[wrapper] EXHAUSTED $MAX_RETRIES retries"
exit 1
