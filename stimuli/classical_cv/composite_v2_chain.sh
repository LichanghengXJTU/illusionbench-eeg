#!/bin/bash
# composite_v2_chain.sh — wait for Phase A's attrs.pkl, then run B → C → D.
set -u
cd /workspace/illusionbench-eeg
. .venv/bin/activate

ATTRS=/workspace/illusionbench-eeg/data/composite_attrs.pkl
PNG_DIR=/workspace/illusionbench-eeg/data/composite_v2_pngs
TAR_DIR=/workspace/illusionbench-eeg/data/composite_v2
LOG_DIR=/workspace/illusionbench-eeg/logs
mkdir -p "$PNG_DIR" "$LOG_DIR"

echo "[chain] $(date) waiting for $ATTRS"
while [ ! -s "$ATTRS" ]; do
  sleep 30
  # Bail if phaseA tmux died with no result
  if ! tmux has-session -t phaseA 2>/dev/null && [ ! -s "$ATTRS" ]; then
    echo "[chain] phaseA tmux dead and no attrs.pkl — aborting"
    exit 1
  fi
done
echo "[chain] $(date) attrs.pkl present ($(stat -c%s $ATTRS) bytes) — starting Phase B"

# === Phase B ===
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -u -m stimuli.classical_cv.composite_v2_parallel \
  --phase b --workers 96 --donors_per_template 3 \
  --tar_src /workspace/illusionbench-eeg/data/HoloFaceIllusion-Bench-EEG-v1.0 \
  --attrs_pkl "$ATTRS" \
  --png_dir "$PNG_DIR" 2>&1 | tee "$LOG_DIR/phaseB.log"
B_RC=${PIPESTATUS[0]}
echo "[chain] Phase B exit=$B_RC"
if [ "$B_RC" -ne 0 ]; then
  echo "[chain] Phase B failed; continuing with whatever was produced"
fi

JSON_CT=$(ls "$PNG_DIR"/*.json 2>/dev/null | wc -l)
echo "[chain] Phase B produced $JSON_CT (T,D) pair JSONs"
if [ "$JSON_CT" -lt 100 ]; then
  echo "[chain] too few outputs ($JSON_CT) — aborting before packaging"
  exit 2
fi

# === Phase C ===
python -u -m stimuli.classical_cv.package_composite_v2 \
  --src "$PNG_DIR" --dst "$TAR_DIR" --pairs_per_shard 500 2>&1 \
  | tee "$LOG_DIR/phaseC.log"
C_RC=${PIPESTATUS[0]}
echo "[chain] Phase C exit=$C_RC"

TAR_CT=$(ls "$TAR_DIR/v2.0"/composite_*.tar 2>/dev/null | wc -l)
echo "[chain] Phase C produced $TAR_CT tar shards"
if [ "$TAR_CT" -lt 1 ]; then
  echo "[chain] no tars — aborting upload"
  exit 3
fi

# === Phase D ===
bash /workspace/illusionbench-eeg/stimuli/classical_cv/upload_composite_v2_stall_retry.sh 2>&1 \
  | tee "$LOG_DIR/phaseD.log"
D_RC=${PIPESTATUS[0]}
echo "[chain] Phase D exit=$D_RC"

echo "[chain] $(date) DONE chain (B=$B_RC C=$C_RC D=$D_RC)"
exit $D_RC
