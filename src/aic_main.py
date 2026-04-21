"""AIC-specific MCAP → LeRobot v3 entry point.

Handles the AIC collector's nested directory structure:

  <runs_root>/
    run_NN_YYYYMMDD_HHMMSS/
      trial_N_scoreN/
        bag/
          *.mcap          ← converted from here
        tags.json         ← success flag, score, parameters

Usage:
  # Convert all trials (including failed)
  python aic_main.py

  # Convert only successful trials
  python aic_main.py --success-only

  # Custom paths
  python aic_main.py \\
    --runs-root ~/aic_community_raw_data/runs \\
    --output    ~/aic_community_raw_data/lerobot \\
    --config    config_hrchung.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from v3_conversion.data_converter import frames_to_episode
from v3_conversion.data_creator import DataCreator
from v3_conversion.hz_checker import validate_from_timestamps
from v3_conversion.mcap_reader import (
    build_extraction_config,
    extract_frames,
    validate_mcap_topics,
)
from v3_conversion.utils import compute_grade

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_hrchung.json"
DEFAULT_RUNS_ROOT = Path.home() / "aic_community_raw_data" / "runs"
DEFAULT_OUTPUT = Path.home() / "aic_community_raw_data" / "lerobot"


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    task_name = cfg.get("task") or cfg.get("task_name")
    if not task_name:
        raise ValueError("Config must have 'task' or 'task_name'")

    return {
        "task_name": task_name,
        "repo_id": cfg.get("repo_id", task_name),
        "robot_type": cfg.get("robot") or cfg.get("robot_type", ""),
        "fps": cfg.get("fps", 20),
        "camera_topic_map": cfg.get("camera_topic_map", {}),
        "joint_names": cfg.get("joint_names", []),
        "state_topic": cfg.get("state_topic", ""),
        "action_topics_map": cfg.get("action_topics_map", {}),
        "task_instruction": cfg.get("task_instruction", []),
        "tags": cfg.get("tags", []),
    }


# ------------------------------------------------------------------
# Trial discovery
# ------------------------------------------------------------------

def _find_trials(runs_root: Path, success_only: bool) -> List[Dict[str, Any]]:
    """Scan runs_root for run_*/trial_*/bag/*.mcap files."""
    trials = []
    for run_dir in sorted(runs_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        for trial_dir in sorted(run_dir.glob("trial_*")):
            if not trial_dir.is_dir():
                continue
            bag_dir = trial_dir / "bag"
            if not bag_dir.is_dir():
                continue
            mcaps = sorted(bag_dir.glob("*.mcap"))
            if not mcaps:
                continue

            tags: dict = {}
            tags_path = trial_dir / "tags.json"
            if tags_path.is_file():
                try:
                    tags = json.loads(tags_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning("Could not read %s: %s", tags_path, e)

            if success_only and not tags.get("success", False):
                logger.info(
                    "Skipping (not success): %s/%s", run_dir.name, trial_dir.name
                )
                continue

            trials.append(
                {
                    "run": run_dir.name,
                    "trial": trial_dir.name,
                    "mcap_path": mcaps[0],
                    "tags": tags,
                }
            )
    return trials


# ------------------------------------------------------------------
# Conversion pipeline
# ------------------------------------------------------------------

def run_aic_conversion(
    runs_root: Path,
    output_root: Path,
    config_path: Path,
    success_only: bool = False,
) -> int:
    cfg = _load_config(config_path)

    task_name = cfg["task_name"]
    repo_id = cfg["repo_id"]
    output_dir = str(output_root / task_name)

    # Build extraction config once — all trials share the same topic layout.
    extraction_config = build_extraction_config(
        detail=cfg,
        fps=cfg["fps"],
        robot_type=cfg["robot_type"],
    )

    task_instruction_list = cfg.get("task_instruction", [])
    task_instruction = task_instruction_list[0] if task_instruction_list else "default_task"

    trials = _find_trials(runs_root, success_only)
    if not trials:
        logger.error("No trials found under %s", runs_root)
        return 1

    logger.info(
        "Starting conversion: task=%s, trials=%d, output=%s",
        task_name, len(trials), output_dir,
    )

    creator: Optional[DataCreator] = None
    converted = 0
    failed = 0
    skipped = 0

    for idx, trial_info in enumerate(trials):
        label = f"{trial_info['run']}/{trial_info['trial']}"
        mcap_path = trial_info["mcap_path"]
        logger.info("[%d/%d] Converting: %s", idx + 1, len(trials), label)

        try:
            # 1. Topic pre-check
            validation = validate_mcap_topics(
                str(mcap_path), extraction_config.topic_map
            )
            if validation["missing_topics"]:
                raise ValueError(
                    f"MCAP topic pre-check failed: missing {validation['missing_topics']}"
                )

            # 2. Extract frames
            frames, timestamps = extract_frames(
                bag_path=str(mcap_path),
                config=extraction_config,
            )

            # 3. Hz validation — 실패 시 skip (count as skipped, not failed)
            hz_result = validate_from_timestamps(
                timestamps=timestamps,
                target_hz=float(extraction_config.fps),
                min_ratio=extraction_config.hz_min_ratio,
                camera_names=extraction_config.camera_names,
            )
            if not hz_result.is_valid:
                logger.warning("  [Hz] SKIP %s — %s", label, hz_result.overall_message)
                skipped += 1
                continue
            logger.info("  [Hz] PASSED: %s", label)

            if not frames:
                raise ValueError(
                    "No frames extracted — all topics present but build_frame() "
                    "returned None for every cycle."
                )

            # 4. Build episode
            episode = frames_to_episode(
                frames=frames,
                action_order=extraction_config.action_order,
                camera_names=extraction_config.camera_names,
                task=task_instruction,
            )

            # 5. Initialize DataCreator on first episode
            if creator is None:
                creator = DataCreator(
                    repo_id=repo_id,
                    root=output_dir,
                    robot_type=extraction_config.robot_type,
                    action_order=extraction_config.action_order,
                    joint_order=extraction_config.joint_order,
                    camera_names=extraction_config.camera_names,
                    fps=extraction_config.fps,
                )

            # 6. Save episode with AIC-specific metadata + QC grade
            tags = trial_info["tags"]
            scoring = tags.get("scoring", {})
            custom_metadata = {
                "run": trial_info["run"],
                "trial": trial_info["trial"],
                "success": str(tags.get("success", "")),
                "policy": tags.get("policy", ""),
                "score_total": str(scoring.get("total", "")),
                "tier_3_score": str(scoring.get("tier_3_score", "")),
                "grade": compute_grade(tags),
                "tags": cfg.get("tags", []),
            }
            creator.convert_episode(episode, custom_metadata=custom_metadata)
            converted += 1
            logger.info("  Converted successfully: %s", label)

        except Exception as e:
            failed += 1
            logger.error(
                "  Failed: %s — %s\n%s", label, e, traceback.format_exc()
            )
            if creator is not None:
                try:
                    creator.recover_dataset_state()
                except Exception as recover_err:
                    logger.error("  Recovery failed: %s", recover_err)
                    creator.dataset = None

    # Finalize
    if creator is not None and creator.dataset is not None:
        try:
            creator.dataset.finalize()
            logger.info("Dataset finalized")
            creator.correct_video_timestamps()
            logger.info("Video timestamps corrected")
            creator.patch_episodes_metadata()
            logger.info("Episode metadata patched")
        except Exception as e:
            logger.error("Finalize failed: %s", e)

    logger.info(
        "Conversion complete: %d converted, %d skipped (Hz), %d failed",
        converted, skipped, failed,
    )

    if converted == 0:
        return 1
    return 2 if failed > 0 else 0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

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
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Root directory of collected runs (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output root for LeRobot dataset (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only convert episodes where tags.json has success=true",
    )
    args = parser.parse_args()

    sys.exit(
        run_aic_conversion(
            runs_root=args.runs_root,
            output_root=args.output,
            config_path=args.config,
            success_only=args.success_only,
        )
    )


if __name__ == "__main__":
    main()
