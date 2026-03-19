#!/bin/bash
set -euo pipefail

echo "============================================"
echo "  v3_conversion - Standalone MCAP Converter"
echo "============================================"

python3 main.py "$@"
