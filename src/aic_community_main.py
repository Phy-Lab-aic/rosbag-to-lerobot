"""AIC community (Format A/B) → LeRobot v3 entry point.

지원 포맷:
  Format B (e2e): run_dir/bag/ + episode/
  Format A (raw/backup): run_dir/trial_N_*/bag/ + episode/

Usage:
  python aic_community_main.py --config config_community.json

  python aic_community_main.py \\
    --config config_community.json \\
    --success-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aic_community_converter import run_conversion

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_community.json"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"JSON config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only convert episodes where tags.json has success=true (overrides config)",
    )
    args = parser.parse_args()

    # --success-only flag overrides config value
    if args.success_only:
        import json
        with open(args.config) as f:
            cfg = json.load(f)
        cfg["success_only"] = True
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name
        try:
            sys.exit(run_conversion(tmp_path))
        finally:
            os.unlink(tmp_path)
    else:
        sys.exit(run_conversion(args.config))


if __name__ == "__main__":
    main()
