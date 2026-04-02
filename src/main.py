"""CLI entry point for standalone MCAP-to-LeRobot conversion.

Replaces ConversionNode (ROS2 Action Server) with a standalone Python script.
Reads JSON config, iterates folders, and produces LeRobot v3 datasets.
"""

import json
import logging
import shutil
import subprocess
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
    extract_frames_iter,
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
        "append": config.get("append", False),
        "cleanup_source_bags": config.get("cleanup_source_bags", False),
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

    # Remove stopped aic_eval container to free disk (distrobox holds fd on deleted files)
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", "aic_eval"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip() == "exited":
            subprocess.run(["docker", "rm", "aic_eval"], capture_output=True, timeout=10)
            logger.info("Removed stopped aic_eval container (freed held disk space)")
    except Exception:
        pass  # docker not available or container doesn't exist — fine

    # Disk space guard — warn if less than 10GB free
    disk_check_path = OUTPUT_PATH if OUTPUT_PATH.exists() else OUTPUT_PATH.parent
    free_gb = shutil.disk_usage(disk_check_path).free / (1024 ** 3)
    if free_gb < 10:
        logger.error(
            "Low disk space: %.1fGB free. At least 10GB recommended. "
            "Clean up Docker containers (docker rm), temp files, or source bags.",
            free_gb,
        )
        return 1
    elif free_gb < 20:
        logger.warning("Disk space is low: %.1fGB free. Consider freeing space.", free_gb)

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

            # 3. Pre-check: all expected topics exist in MCAP
            validation = validate_mcap_topics(str(mcap_path), config.topic_map)
            if validation["missing_topics"]:
                raise ValueError(
                    f"MCAP topic pre-check failed [folder={folder_name}]: "
                    f"missing {validation['missing_topics']}"
                )

            # 5. Two-pass streaming conversion (avoids OOM on large bags)
            #    Pass 1: collect obs/action (small arrays only)
            #    Pass 2: stream images frame-by-frame to DataCreator
            import numpy as np

            task_instruction = "default_task"
            ti = metadata.get("task_instruction")
            if ti and isinstance(ti, list) and len(ti) > 0 and ti[0]:
                task_instruction = ti[0]

            # --- Pass 1: obs/action only ---
            obs_list = []
            action_lists = {key: [] for key in config.action_order}
            first_frame_images = None

            frame_count = 0
            for frame in extract_frames_iter(
                bag_path=str(mcap_path), config=config,
            ):
                obs_list.append(np.asarray(frame["obs"], dtype=np.float32))
                for key in config.action_order:
                    action_lists[key].append(
                        np.asarray(frame["action"][key], dtype=np.float32)
                    )
                if first_frame_images is None and config.camera_names:
                    import cv2
                    first_frame_images = {
                        cam: cv2.resize(frame["images"][cam], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                        for cam in config.camera_names
                        if cam in frame["images"]
                    }
                frame_count += 1

            if frame_count == 0:
                raise ValueError(
                    f"No frames extracted [folder={folder_name}] from {mcap_path}."
                )

            logger.info("  Pass 1 done: %d frames (obs/action collected)", frame_count)

            # Collect obs/action arrays
            obs_array = np.stack(obs_list, axis=0)
            actions_by_key = {}
            for key in config.action_order:
                actions_by_key[key] = np.stack(action_lists[key], axis=0).astype(np.float32)
            del obs_list, action_lists

            # Concatenate action dims
            all_actions = np.concatenate(
                [actions_by_key[key] for key in config.action_order], axis=-1
            ).astype(np.float32)

            # --- Initialize DataCreator ---
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

                append_mode = cfg.get("append", False)
                dataset_root = Path(output_root)
                local_exists = (dataset_root / "meta" / "info.json").is_file()

                if append_mode and not local_exists and "/" in repo_id:
                    # Pull existing dataset from HuggingFace to local path
                    try:
                        logger.info("Append mode: pulling existing dataset from Hub: %s", repo_id)
                        from lerobot.datasets.lerobot_dataset import LeRobotDataset
                        # Download to default cache, then copy to output_root
                        tmp_ds = LeRobotDataset(repo_id=repo_id)
                        cached_root = Path(tmp_ds.root)
                        dataset_root.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(str(cached_root), str(dataset_root))
                        del tmp_ds
                        local_exists = True
                        logger.info("Pulled %s to %s", repo_id, dataset_root)
                    except Exception as e:
                        logger.warning("No existing dataset on Hub, creating new: %s", e)

                if append_mode and local_exists:
                    logger.info("Append mode: loading existing dataset (%d episodes)",
                                creator.dataset.meta.total_episodes if creator.dataset else 0)
                    creator.load_dataset()
                    logger.info("Loaded existing dataset with %d episodes",
                                creator.dataset.meta.total_episodes)
                else:
                    # Build a minimal episode dict for create_dataset shape inference
                    dummy_episode = {
                        "obs": obs_array[:1],
                        "images": {cam: [img] for cam, img in (first_frame_images or {}).items()},
                        "task": task_instruction,
                    }
                    for key in config.action_order:
                        dummy_episode[key] = actions_by_key[key][:1]
                    creator.create_dataset(dummy_episode)

            if creator.dataset.episode_buffer is None or "size" not in creator.dataset.episode_buffer:
                creator.dataset.episode_buffer = creator.dataset.create_episode_buffer()

            # --- Pass 2: stream images + write frames ---
            logger.info("  Pass 2: streaming %d frames to dataset...", frame_count)
            frame_idx = 0
            for frame in extract_frames_iter(
                bag_path=str(mcap_path), config=config,
            ):
                lerobot_frame = {
                    "observation.state": obs_array[frame_idx],
                    "action": all_actions[frame_idx],
                }
                for cam in config.camera_names:
                    if cam in frame["images"]:
                        import cv2
                        img = frame["images"][cam]
                        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                        lerobot_frame[f"observation.images.{cam}"] = img
                lerobot_frame["task"] = task_instruction
                creator.dataset.add_frame(lerobot_frame)

                frame_idx += 1
                if frame_idx % 500 == 0:
                    logger.info("    Written %d/%d frames", frame_idx, frame_count)

            creator.dataset.save_episode()
            logger.info("  Episode saved: %d frames", frame_idx)
            del obs_array, all_actions, actions_by_key

            # Skip the old convert_episode path
            episode = None

            # 8. Store custom metadata (episode already saved via streaming)
            custom_metadata = {
                "Serial_number": folder_name,
                "tags": metadata.get("tags", []),
                "grade": "",
            }
            creator._episode_custom_metadata.append(custom_metadata)

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
        push_success = False
        if "/" in repo_id:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                ds = LeRobotDataset(repo_id=repo_id, root=output_root)
                ds.push_to_hub(tags=cfg.get("tags") or None, push_videos=True)
                logger.info("Pushed dataset to HuggingFace Hub: %s", repo_id)
                push_success = True
            except Exception as e:
                err_msg = str(e)
                # RepoCard/Jinja2 error means data was already pushed successfully
                if "Jinja2" in err_msg or "RepoCard" in err_msg:
                    logger.warning("Dataset pushed but RepoCard skipped (install Jinja2 to fix): %s", e)
                    push_success = True
                else:
                    logger.error("Failed to push to Hub: %s", e)
        else:
            logger.warning(
                "Skipping push_to_hub: repo_id '%s' is not in 'namespace/name' format. "
                "Set 'repo_id' in config.json (e.g. 'myuser/mydataset') to enable upload.",
                repo_id,
            )

        # --- Cleanup ---
        # 1. Remove local dataset after successful upload (skip in append mode)
        append_mode = cfg.get("append", False)
        if push_success and not append_mode and Path(output_root).exists():
            try:
                shutil.rmtree(output_root)
                logger.info("Cleaned up local dataset: %s", output_root)
            except Exception as e:
                logger.warning("Failed to clean up local dataset: %s", e)
        elif push_success and append_mode:
            logger.info("Append mode: keeping local dataset for future appends: %s", output_root)

        # 2. Remove source bags after successful conversion (opt-in)
        if cfg.get("cleanup_source_bags") and converted_count > 0 and failed_count == 0:
            for folder_name in folders:
                bag_dir = INPUT_PATH / folder_name
                if bag_dir.exists():
                    try:
                        shutil.rmtree(bag_dir)
                        logger.info("Cleaned up source bag: %s", bag_dir)
                    except Exception as e:
                        logger.warning("Failed to clean up source bag %s: %s", bag_dir, e)

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
