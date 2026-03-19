"""CLI entry point for v3_conversion."""

import logging
import sys

from v3_conversion.constants import CONFIG_PATH
from v3_conversion.converter import run_conversion

DEFAULT_CONFIG = CONFIG_PATH


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CONFIG)
    exit_code = run_conversion(config_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
