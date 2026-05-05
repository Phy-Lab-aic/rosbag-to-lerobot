#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# `lerobot` env carries CUDA torch + lerobot package.
CONDA_BASE="${CONDA_BASE:-/home/weed/miniconda3}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# torchcodec links against bundled NVIDIA wheels — expose them on the loader path.
NV_BASE="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
if [ -d "${NV_BASE}" ]; then
    NV_LIBS=$(find "${NV_BASE}" -maxdepth 3 -type d -name lib | tr '\n' ':')
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
fi

python3 "${SCRIPT_DIR}/smoke_train_act.py" "$@"
