"""Source-file loaders for AIC run/trial/episode metadata."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml


_TAG_FIELDS = (
    "schema_version",
    "trial",
    "success",
    "early_terminated",
    "early_term_source",
)

_CATEGORY_COLUMNS = {
    "contacts": ("score_contacts", "score_contacts_message"),
    "duration": ("score_duration", "score_duration_message"),
    "insertion force": (
        "score_insertion_force",
        "score_insertion_force_message",
    ),
    "trajectory efficiency": (
        "score_traj_efficiency",
        "score_traj_efficiency_message",
    ),
    "trajectory smoothness": (
        "score_traj_smoothness",
        "score_traj_smoothness_message",
    ),
}


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


def load_tags(trial_dir: Path) -> Dict[str, Any]:
    """Read tags.json and extract only the fields the meta schema uses."""
    path = trial_dir / "tags.json"
    defaults: Dict[str, Any] = {
        "schema_version": "",
        "trial": -1,
        "success": False,
        "early_terminated": False,
        "early_term_source": "",
    }
    if not path.is_file():
        return defaults
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: raw.get(k, defaults[k]) for k in _TAG_FIELDS}


def load_scoring_yaml(trial_dir: Path, trial_key: str) -> Dict[str, Any]:
    """Read scoring.yaml and flatten it into the meta/aic/scoring.parquet shape."""
    path = trial_dir / "scoring.yaml"
    result: Dict[str, Any] = {
        "score_total": float("nan"),
        "score_tier1": float("nan"),
        "score_tier2": float("nan"),
        "score_tier3": float("nan"),
    }
    for score_col, msg_col in _CATEGORY_COLUMNS.values():
        result[score_col] = float("nan")
        result[msg_col] = ""

    if not path.is_file():
        return result

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    trial = raw.get(trial_key) or {}

    result["score_total"] = float(trial.get("total", float("nan")))
    for tier in (1, 2, 3):
        tier_block = trial.get(f"tier_{tier}", {})
        result[f"score_tier{tier}"] = float(tier_block.get("score", float("nan")))

    categories = (trial.get("tier_2") or {}).get("categories") or {}
    for cat_key, (score_col, msg_col) in _CATEGORY_COLUMNS.items():
        cat = categories.get(cat_key) or {}
        result[score_col] = float(cat.get("score", float("nan")))
        result[msg_col] = str(cat.get("message", ""))

    return result
