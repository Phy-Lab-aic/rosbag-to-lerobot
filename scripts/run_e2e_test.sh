#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# Activate the conda env that has lerobot + datasets + mcap deps
CONDA_BASE="${CONDA_BASE:-/home/weed/miniconda3}"
CONDA_ENV="${CONDA_ENV:-rosbag2lerobot}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Isolate from system / user-site packages
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/lerobot/src"

# E2E test paths
export INPUT_PATH="${INPUT_PATH:-/home/weed/aic_community_e2e}"
export OUTPUT_PATH="${OUTPUT_PATH:-/home/weed/aic_results/lerobot_output_e2e_test}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/src/config_e2e_test.json}"

mkdir -p "${OUTPUT_PATH}"

echo "============================================"
echo "  E2E Test: 10 community runs -> LeRobot v3"
echo "============================================"
echo "INPUT_PATH : ${INPUT_PATH}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CONFIG     : ${CONFIG_PATH}"
echo ""

python3 "${REPO_ROOT}/src/main.py" "${CONFIG_PATH}" \
    --input-dir "${INPUT_PATH}" \
    --output-dir "${OUTPUT_PATH}" \
    "$@"
