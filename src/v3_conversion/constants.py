"""Path constants for the standalone conversion pipeline."""

import os
from pathlib import Path

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/home/weed/aic_results/cheatcode_dataset"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/home/weed/aic_results/lerobot_output"))
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
