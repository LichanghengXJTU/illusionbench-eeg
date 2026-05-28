#!/bin/bash
# composite_v2_post_b.sh — wait for Phase B DONE, then run C → D.
set -u
cd /workspace/illusionbench-eeg
. .venv/bin/activate

PNG_DIR=/workspace/illusionbench-eeg/data/composite_v2_pngs
TAR_DIR=/workspace/illusionbench-eeg/data/composite_v2
LOG_DIR=/workspace/illusionbench-eeg/logs
mkdir -p "$LOG_DIR"

echo "[post] $(date) waiting for Phase B to finish..."
# Wait until no composite_v2_parallel proc AND log contains DONE line
while true; do
  alive=$(pgrep -f composite_v2_parallel | wc -l)
  if [ "$alive" -eq 0 ]; then
    if grep -q "phase B] DONE" /workspace/illusionbench-eeg/logs/phaseB32.log 2>/dev/null; then
      echo "[post] $(date) Phase B finished (DONE in log)"
      break
    else
      echo "[post] $(date) procs gone but no DONE in log — assuming crash, breaking anyway"
      break
    fi
  fi
  sleep 30
done

# Use `find` (not `ls *.json`) — at >1k files the glob overflows ARG_MAX and silently returns 0.
JSON_CT=$(find "$PNG_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
echo "[post] Phase B produced $JSON_CT (T,D) pair JSONs"
if [ "$JSON_CT" -lt 100 ]; then
  echo "[post] too few outputs ($JSON_CT) — aborting"
  exit 2
fi

# === Phase C ===
echo "[post] $(date) starting Phase C"
python -u -m stimuli.classical_cv.package_composite_v2 \
  --src "$PNG_DIR" --dst "$TAR_DIR" --pairs_per_shard 500 2>&1 \
  | tee "$LOG_DIR/phaseC.log"
C_RC=${PIPESTATUS[0]}
echo "[post] Phase C exit=$C_RC"

TAR_CT=$(find "$TAR_DIR/v2.0" -maxdepth 1 -name "composite_*.tar" 2>/dev/null | wc -l)
echo "[post] Phase C produced $TAR_CT tar shards"
if [ "$TAR_CT" -lt 1 ]; then
  echo "[post] no tars — aborting upload"
  exit 3
fi

# === Phase D ===
echo "[post] $(date) starting Phase D upload"
bash /workspace/illusionbench-eeg/stimuli/classical_cv/upload_composite_v2_stall_retry.sh 2>&1 \
  | tee "$LOG_DIR/phaseD.log"
D_RC=${PIPESTATUS[0]}
echo "[post] Phase D exit=$D_RC"

echo "[post] $(date) DONE (C=$C_RC D=$D_RC)"
exit $D_RC
