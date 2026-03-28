#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# Isolate conda env from system/ROS2/user-site packages
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/lerobot/src"

echo "============================================"
echo "  v3_conversion - Standalone MCAP Converter"
echo "============================================"
echo "INPUT_PATH : ${INPUT_PATH:-<default in constants.py>}"
echo "OUTPUT_PATH: ${OUTPUT_PATH:-<default in constants.py>}"
echo ""

python3 "${REPO_ROOT}/src/main.py" "$@"
