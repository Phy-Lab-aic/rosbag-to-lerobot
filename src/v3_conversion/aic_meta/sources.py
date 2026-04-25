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

_METADATA_MAP = {
    "cable_type": "cable_type",
    "cable_name": "cable_name",
    "plug_type": "plug_type",
    "plug_name": "plug_name",
    "port_type": "port_type",
    "port_name": "port_name",
    "target_module": "target_module",
    "success": "success",
    "early_terminated": "early_terminated",
    "early_term_source": "early_term_source",
    "num_steps": "num_steps",
    "duration_sec": "duration_sec",
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


def load_episode_metadata(episode_dir: Path) -> Dict[str, Any]:
    """Read episode/metadata.json into a flat subset for the meta schema."""
    path = episode_dir / "metadata.json"
    defaults: Dict[str, Any] = {
        "cable_type": "",
        "cable_name": "",
        "plug_type": "",
        "plug_name": "",
        "port_type": "",
        "port_name": "",
        "target_module": "",
        "success": False,
        "early_terminated": False,
        "early_term_source": "",
        "num_steps": 0,
        "duration_sec": 0.0,
        "plug_port_distance_init": float("nan"),
    }
    if not path.is_file():
        return defaults

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    result = dict(defaults)
    for src, dst in _METADATA_MAP.items():
        if src in raw:
            result[dst] = raw[src]
    if "plug_port_distance" in raw:
        result["plug_port_distance_init"] = float(raw["plug_port_distance"])
    return result


def _is_rail_key(key: str) -> bool:
    return "rail_" in key


def load_scene_from_config(run_dir: Path, trial_key: str) -> Dict[str, Any]:
    """Read config.yaml and extract rails + cable initial pose (gripper frame)."""
    path = run_dir / "config.yaml"
    result: Dict[str, Any] = {
        "scene_rails": [],
        "initial_plug_pose_rel_gripper": [0.0] * 6,
    }
    if not path.is_file():
        return result

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    trial = (raw.get("trials") or {}).get(trial_key) or {}
    scene = trial.get("scene") or {}

    task_board = scene.get("task_board") or {}
    rails: list[Dict[str, Any]] = []
    for name, val in task_board.items():
        if not _is_rail_key(name) or not isinstance(val, dict):
            continue
        rails.append(
            {
                "name": name,
                "entity_present": bool(val.get("entity_present", False)),
                "entity_name": str(val.get("entity_name", "")),
            }
        )
    result["scene_rails"] = rails

    cables = scene.get("cables") or {}
    cable_0 = cables.get("cable_0") if isinstance(cables, dict) else None
    if isinstance(cable_0, dict):
        pose = cable_0.get("pose") or {}
        offset = pose.get("gripper_offset") or {}
        result["initial_plug_pose_rel_gripper"] = [
            float(offset.get("x", 0.0)),
            float(offset.get("y", 0.0)),
            float(offset.get("z", 0.0)),
            float(pose.get("roll", 0.0)),
            float(pose.get("pitch", 0.0)),
            float(pose.get("yaw", 0.0)),
        ]

    return result


def load_validation_status(run_dir: Path) -> Dict[str, Any]:
    """Return whether validation.json permits conversion for a run."""
    path = run_dir / "validation.json"
    if not path.is_file():
        return {"passed": False, "reason": "validation.json missing"}

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        return {"passed": False, "reason": "validation.json invalid json"}
    except OSError:
        return {"passed": False, "reason": "validation.json unreadable"}

    if not isinstance(raw, dict):
        return {"passed": False, "reason": "validation.json invalid: expected object"}

    for key in ("passed_count", "total_count", "checks"):
        if key not in raw:
            return {"passed": False, "reason": f"validation.json invalid: missing {key}"}

    checks = raw["checks"]
    if not isinstance(checks, list):
        return {"passed": False, "reason": "validation.json invalid: checks must be list"}

    for check in checks:
        if isinstance(check, dict) and check.get("passed") is False:
            name = str(check.get("name", "unnamed"))
            return {"passed": False, "reason": f"validation.json failed check: {name}"}

    passed_count = raw["passed_count"]
    total_count = raw["total_count"]
    if passed_count != total_count:
        return {
            "passed": False,
            "reason": f"validation.json passed_count {passed_count} != total_count {total_count}",
        }

    return {"passed": True, "reason": ""}
