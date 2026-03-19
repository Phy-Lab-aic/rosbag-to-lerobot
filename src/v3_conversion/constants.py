"""Path constants for the standalone conversion pipeline."""

from pathlib import Path

INPUT_PATH = Path("/data/raw")
OUTPUT_PATH = Path("/data/lerobot")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
