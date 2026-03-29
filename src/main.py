"""CLI entry point for standalone MCAP-to-LeRobot conversion.

Replaces ConversionNode (ROS2 Action Server) with a standalone Python script.
Reads JSON config, iterates folders, and produces LeRobot v3 datasets.
"""

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from v3_conversion.constants import CONFIG_PATH, INPUT_PATH, OUTPUT_PATH
from v3_conversion.data_converter import frames_to_episode
from v3_conversion.data_creator import DataCreator
from v3_conversion.hz_checker import validate_from_timestamps
from v3_conversion.mcap_reader import (
    build_extraction_config,
    extract_frames,
    validate_mcap_topics,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = CONFIG_PATH


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    """Load and validate JSON config file.

    Config-level fields (camera_topic_map, joint_names, state_topic,
    action_topics_map) serve as defaults when metacard.json is absent.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    task_name = config.get("task") or config.get("task_name")
    if not task_name:
        raise ValueError("Config must have 'task' or 'task_name'")

    # repo_id for HuggingFace Hub (must be "namespace/dataset-name" format)
    repo_id = config.get("repo_id", "")

    folders = config.get("folders", [])
    if folders == "all":
        # Scan INPUT_PATH for all subdirectories containing .mcap files
        if INPUT_PATH.is_dir():
            folders = sorted([
                d.name for d in INPUT_PATH.iterdir()
                if d.is_dir() and any(d.glob("*.mcap"))
            ])
        else:
            folders = []
    elif not isinstance(folders, list):
        raise ValueError(
            f"'folders' must be a list of folder names or 'all', got: {type(folders).__name__}"
        )

    robot_type = config.get("robot") or config.get("robot_type") or ""
    fps = config.get("fps", None)

    return {
        "task_name": task_name,
        "repo_id": repo_id,
        "folders": folders,
        "robot_type": robot_type,
        "fps": fps,
        # Config-level defaults used when metacard.json is missing
        "camera_topic_map": config.get("camera_topic_map", {}),
        "joint_names": config.get("joint_names", []),
        "state_topic": config.get("state_topic", ""),
        "action_topics_map": config.get("action_topics_map", {}),
        "task_instruction": config.get("task_instruction", []),
        "tags": config.get("tags", []),
    }


# ------------------------------------------------------------------
# Metacard loading
# ------------------------------------------------------------------

def _load_metacard(
    folder_name: str,
    config_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and parse metacard.json from a raw folder.

    When metacard.json is absent, falls back to *config_defaults* (the
    config-level topic/joint fields).  Any field present in the metacard
    takes precedence over the config default.
    """
    defaults = config_defaults or {}
    metacard_path = INPUT_PATH / folder_name / "metacard.json"

    if metacard_path.is_file():
        with metacard_path.open("r", encoding="utf-8") as f:
            metacard = json.load(f)
    else:
        logger.info(
            "  metacard.json not found for '%s', using config defaults",
            folder_name,
        )
        metacard = {}

    ti = metacard.get("task_instruction", defaults.get("task_instruction", []))
    if not isinstance(ti, list):
        ti = [str(ti)] if ti else []

    return {
        "folder_dir": folder_name,
        "fps": int(metacard.get("fps", defaults.get("fps") or 30)),
        "robot_type": str(metacard.get("robot_type", defaults.get("robot_type", ""))),
        "task_instruction": ti,
        "tags": metacard.get("tags", defaults.get("tags", [])),
        "camera_topic_map": metacard.get("camera_topic_map", defaults.get("camera_topic_map", {})),
        "joint_topic_map": metacard.get("joint_topic_map", defaults.get("joint_topic_map", {})),
        "joint_names": metacard.get("joint_names", defaults.get("joint_names", [])),
        "action_topics_map": metacard.get("action_topics_map", defaults.get("action_topics_map", {})),
        "state_topic": metacard.get("state_topic", defaults.get("state_topic", "")),
    }


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------

def _find_mcap(folder_name: str) -> Path:
    """Locate the MCAP file inside a raw folder.

    Tries ``<folder>/<folder>_0.mcap`` first, then falls back to the
    first ``*.mcap`` found in the directory.
    """
    canonical = INPUT_PATH / folder_name / f"{folder_name}_0.mcap"
    if canonical.is_file():
        return canonical

    folder_dir = INPUT_PATH / folder_name
    mcaps = sorted(folder_dir.glob("*.mcap"))
    if mcaps:
        return mcaps[0]

    raise FileNotFoundError(f"No MCAP file found in {folder_dir}")


def _prepare_config(
    folder_name: str,
    metadata: Dict[str, Any],
    robot_type_override: str = "",
    fps_override: Optional[int] = None,
):
    """Validate metadata and build extraction config.

    Required fields: camera_topic_map, joint_names, action_topics_map,
    state_topic.  When any of these are empty the folder is skipped.
    """
    mcap_path = _find_mcap(folder_name)

    camera_topic_map = metadata.get("camera_topic_map")
    if not isinstance(camera_topic_map, dict):
        camera_topic_map = {}
    if not camera_topic_map:
        logger.info("  camera_topic_map is empty [folder=%s], skipping cameras", folder_name)

    joint_names = metadata.get("joint_names")
    if not isinstance(joint_names, list) or not joint_names:
        raise ValueError(
            f"joint_names is empty or missing [folder={folder_name}]. "
            "Provide it in metacard.json or config.json."
        )

    action_topics_map = metadata.get("action_topics_map")
    if not isinstance(action_topics_map, dict) or not action_topics_map:
        raise ValueError(
            f"action_topics_map is empty or missing [folder={folder_name}]. "
            "Provide it in metacard.json or config.json."
        )

    state_topic = metadata.get("state_topic")
    if not isinstance(state_topic, str) or not state_topic.strip():
        raise ValueError(
            f"state_topic is empty or missing [folder={folder_name}]. "
            "Provide it in metacard.json or config.json."
        )

    fps = fps_override if fps_override is not None else int(metadata.get("fps", 30))
    robot_type = robot_type_override if robot_type_override else str(metadata.get("robot_type", "")).strip()

    config = build_extraction_config(
        detail=metadata,
        fps=fps,
        robot_type=robot_type,
    )

    # Validate: observation and action dimension count
    obs_names = list(config.joint_order.get("obs", []))
    action_cfg = config.joint_order.get("action", {})
    action_names = []
    for key in config.action_order:
        action_names += action_cfg.get(key, [])
    if set(obs_names) != set(action_names):
        logger.warning(
            "  obs/action joint names differ [folder=%s]: "
            "observation.state(%d)=%s, action(%d)=%s",
            folder_name, len(obs_names), obs_names,
            len(action_names), action_names,
        )

    return mcap_path, config


# ------------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------------

def run_conversion(config_path: str) -> int:
    """Run the full conversion pipeline.

    Returns exit code: 0=success, 1=all failed, 2=partial success.
    """
    cfg = _load_config(config_path)
    task_name = cfg["task_name"]
    repo_id = cfg["repo_id"] or task_name
    folders = cfg["folders"]
    robot_type = cfg["robot_type"]
    fps_override = cfg["fps"]

    # Config-level defaults for metacard fallback
    config_defaults = {
        "robot_type": robot_type,
        "fps": fps_override,
        "camera_topic_map": cfg.get("camera_topic_map", {}),
        "joint_names": cfg.get("joint_names", []),
        "state_topic": cfg.get("state_topic", ""),
        "action_topics_map": cfg.get("action_topics_map", {}),
        "task_instruction": cfg.get("task_instruction", []),
        "tags": cfg.get("tags", []),
    }

    if not folders:
        logger.error("No folders to convert")
        return 1

    logger.info("Starting conversion: task=%s, folders=%d", task_name, len(folders))

    output_root = str(OUTPUT_PATH / task_name)
    creator: Optional[DataCreator] = None
    converted_count = 0
    failed_count = 0
    failed_folders: List[str] = []

    for idx, folder_name in enumerate(folders):
        logger.info("[%d/%d] Converting: %s", idx + 1, len(folders), folder_name)

        try:
            # 1. Load metacard (falls back to config defaults)
            metadata = _load_metacard(folder_name, config_defaults)

            # 2. Validate and build config
            mcap_path, config = _prepare_config(
                folder_name, metadata, robot_type, fps_override
            )

            # 3. Initialize DataCreator on first successful config
            if creator is None:
                creator = DataCreator(
                    repo_id=repo_id,
                    root=output_root,
                    robot_type=config.robot_type,
                    action_order=config.action_order,
                    joint_order=config.joint_order,
                    camera_names=config.camera_names,
                    fps=config.fps,
                )

            # 4. Pre-check: all expected topics exist in MCAP
            validation = validate_mcap_topics(str(mcap_path), config.topic_map)
            if validation["missing_topics"]:
                raise ValueError(
                    f"MCAP topic pre-check failed [folder={folder_name}]: "
                    f"missing {validation['missing_topics']}"
                )

            # 5. Extract frames
            frames, timestamps = extract_frames(
                bag_path=str(mcap_path), config=config,
            )

            # 6. Hz validation
            logger.info("  [Hz] validating %s (target=%dHz)", folder_name, config.fps)
            hz_result = validate_from_timestamps(
                timestamps=timestamps,
                target_hz=float(config.fps),
                min_ratio=config.hz_min_ratio,
                camera_names=config.camera_names,
            )
            if not hz_result.is_valid:
                raise ValueError(
                    f"[folder={folder_name}] {hz_result.overall_message}"
                )
            logger.info("  [Hz] PASSED: %s", folder_name)

            if not frames:
                raise ValueError(
                    f"No frames extracted [folder={folder_name}] from {mcap_path}. "
                    f"All expected topics present but build_frame() returned None "
                    f"for every cycle — likely cause: incomplete action data "
                    f"(missing leader topics or action key mismatch)."
                )

            # 7. Transform to episode (with task_instruction extraction)
            task_instruction = "default_task"
            ti = metadata.get("task_instruction")
            if ti and isinstance(ti, list) and len(ti) > 0 and ti[0]:
                task_instruction = ti[0]

            episode = frames_to_episode(
                frames=frames,
                action_order=config.action_order,
                camera_names=config.camera_names,
                task=task_instruction,
            )

            # 8. Convert episode with custom metadata
            custom_metadata = {
                "Serial_number": folder_name,
                "tags": metadata.get("tags", []),
                "grade": "",
            }
            creator.convert_episode(episode, custom_metadata=custom_metadata)

            converted_count += 1
            logger.info("  Converted successfully: %s", folder_name)

        except Exception as e:
            failed_count += 1
            failed_folders.append(folder_name)
            logger.error("  Failed to convert %s: %s\n%s", folder_name, e, traceback.format_exc())
            # Error recovery: reload dataset state so next folder can proceed
            if creator is not None:
                try:
                    creator.recover_dataset_state()
                except Exception as recover_err:
                    logger.error("  Recovery failed: %s", recover_err)
                    creator.dataset = None

    # Finalize dataset
    if creator is not None and creator.dataset is not None:
        try:
            creator.dataset.finalize()
            logger.info("Dataset finalized")
            creator.correct_video_timestamps()
            logger.info("Video timestamps corrected")
            creator.patch_episodes_metadata()
            logger.info("Episode custom metadata patched")
        except Exception as e:
            logger.error("Failed to finalize dataset: %s", e)

        # Push to HuggingFace Hub if repo_id has namespace format
        if "/" in repo_id:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                ds = LeRobotDataset(repo_id=repo_id, root=output_root)
                ds.push_to_hub(tags=cfg.get("tags") or None, push_videos=True)
                logger.info("Pushed dataset to HuggingFace Hub: %s", repo_id)
            except Exception as e:
                logger.error("Failed to push to Hub: %s", e)
        else:
            logger.warning(
                "Skipping push_to_hub: repo_id '%s' is not in 'namespace/name' format. "
                "Set 'repo_id' in config.json (e.g. 'myuser/mydataset') to enable upload.",
                repo_id,
            )

    # Summary
    logger.info(
        "Conversion complete: %d converted, %d failed", converted_count, failed_count
    )
    if failed_folders:
        logger.info("Failed folders: %s", failed_folders)

    if converted_count == 0:
        return 1
    if failed_count > 0:
        return 2
    return 0


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
