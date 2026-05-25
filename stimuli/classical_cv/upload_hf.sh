#!/usr/bin/env bash
# upload_hf.sh — push HoloFaceIllusion-Bench-EEG v1.0 to Hugging Face.
#
# Usage (on server):
#   source /workspace/.secrets/hf.env   # must define HF_TOKEN=hf_xxx
#   HF_USERNAME=<your_username> bash /workspace/illusionbench-eeg/stimuli/classical_cv/upload_hf.sh
#
# Or one-liner:
#   source /workspace/.secrets/hf.env && \
#   HF_USERNAME=<your_username> bash stimuli/classical_cv/upload_hf.sh
#
# What it does:
#   1. Verifies HF_TOKEN + HF_USERNAME are set
#   2. Creates the dataset repo on HF (idempotent — skips if exists)
#   3. Uploads all 70 .tar shards + README + manifest via hf_transfer
#      (parallel multi-stream, ~30-90 min on RunPod ~200 Mbps)
#   4. Prints the final URL

set -euo pipefail

REPO_NAME="HoloFaceIllusion-Bench-EEG"
DATA_DIR="/workspace/illusionbench-eeg/data/HoloFaceIllusion-Bench-EEG-v1.0"
VENV_PY="/workspace/illusionbench-eeg/.venv/bin/python"
CLI="/workspace/illusionbench-eeg/.venv/bin/huggingface-cli"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set."
  echo "  Run: source /workspace/.secrets/hf.env"
  exit 1
fi
if [[ -z "${HF_USERNAME:-}" ]]; then
  echo "ERROR: HF_USERNAME not set."
  echo "  Run: HF_USERNAME=your_handle bash $0"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir not found: $DATA_DIR"
  exit 1
fi

REPO_ID="${HF_USERNAME}/${REPO_NAME}"
echo "[plan] uploading to:  https://huggingface.co/datasets/${REPO_ID}"
echo "[plan] data:          ${DATA_DIR}"
echo "[plan] total size:    $(du -sh "$DATA_DIR" | cut -f1)"
echo "[plan] tar shard cnt: $(ls "$DATA_DIR"/*.tar 2>/dev/null | wc -l)"
echo

# Step 1 — create repo (idempotent)
echo "[step 1/2] creating dataset repo (idempotent)..."
$VENV_PY <<PYEOF
from huggingface_hub import create_repo
import os
try:
    url = create_repo(
        repo_id="${REPO_ID}",
        repo_type="dataset",
        private=False,
        token=os.environ["HF_TOKEN"],
        exist_ok=True,
    )
    print(f"  OK: {url}")
except Exception as e:
    print(f"  ERR: {e}")
    raise SystemExit(1)
PYEOF

# Step 2 — upload everything
echo "[step 2/2] uploading via hf_transfer (parallel)..."
HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="${HF_TOKEN}" \
  $CLI upload "${REPO_ID}" "${DATA_DIR}" . \
    --repo-type=dataset \
    --commit-message="v1.0 release: 26,317 Thatcher identities × 4 conditions"

echo
echo "[done] dataset live at:"
echo "         https://huggingface.co/datasets/${REPO_ID}"
echo
echo "[NEXT] Revoke the HF token at:"
echo "         https://huggingface.co/settings/tokens"
echo "       (security hygiene — the token is in your shell history + the env file)"
