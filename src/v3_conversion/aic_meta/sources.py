"""Source-file loaders for AIC run/trial/episode metadata."""

from pathlib import Path
from typing import Any, Dict


def load_run_meta(run_dir: Path) -> Dict[str, Any]:
    """Read policy.txt and seed.txt from a run directory.

    Missing files yield empty / sentinel values; caller decides how to treat them.
    """
    policy_path = run_dir / "policy.txt"
    seed_path = run_dir / "seed.txt"

    policy = policy_path.read_text().strip() if policy_path.is_file() else ""

    seed_raw = seed_path.read_text().strip() if seed_path.is_file() else ""
    try:
        seed = int(seed_raw) if seed_raw else -1
    except ValueError:
        seed = -1

    return {"policy": policy, "seed": seed}
