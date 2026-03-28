#!/bin/bash
set -euo pipefail

ENV_NAME="${1:-rosbag2lerobot}"

echo "============================================"
echo "  Conda Environment Setup: ${ENV_NAME}"
echo "============================================"

# Isolate conda env from system/ROS2/user-site packages
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Create conda environment with pip included
conda create -n "${ENV_NAME}" python=3.12 pip -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# System-level deps via conda-forge
conda install -c conda-forge ffmpeg -y

# Python dependencies (matching Dockerfile.conversion)
# Use python -m pip to ensure conda env's pip is used
python -m pip install --no-cache-dir \
    "numpy==2.2.6" \
    "opencv-python-headless>=4.9.0,<4.13.0" \
    "av>=15.0.0,<16.0.0" \
    "pyarrow>=15.0.0" \
    "imageio>=2.34.0,<3.0.0" \
    "pillow>=10.0.0" \
    "pyyaml>=6.0" \
    "pandas>=2.0.0" \
    "mcap>=0.0.10" \
    "mcap-ros2-support>=0.5.0" \
    datasets requests tqdm packaging fsspec

# torch-stub (avoids pulling full PyTorch ~731MB)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

python -m pip install --no-cache-dir "${REPO_ROOT}/docker/torch-stub"

# Install lerobot (no-deps to avoid pulling PyTorch)
python -m pip install --no-cache-dir --no-deps -e "${REPO_ROOT}/lerobot"

echo ""
echo "Done! Activate with:  conda activate ${ENV_NAME}"
echo "Run conversion with:  bash scripts/run_convert.sh"
