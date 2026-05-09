"""CLI entry point for standalone MCAP-to-LeRobot conversion.

Replaces ConversionNode (ROS2 Action Server) with a standalone Python script.
Reads JSON config, iterates folders, and produces LeRobot v3 datasets.

Supports two modes:
  - Convert:  python main.py config.json --input-dir /bags --output-dir /out
  - Merge:    python main.py config.json --input-dir /bags --output-dir /existing/dataset --merge
"""

import argparse
import gc
import json
import logging
import sys
import traceback
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pyarrow.parquet as pq

from v3_conversion.action_shift import apply_one_step_shift
from v3_conversion.aic_meta.scoring_mcap import (
    extract_insertion_event,
    extract_scoring_tf_snapshots,
)
from v3_conversion.aic_meta.pose_commands import extract_pose_commands
from v3_conversion.aic_meta.pose_labels import extract_pose_labels
from v3_conversion.aic_meta.sources import (
    load_episode_metadata,
    load_run_meta,
    load_scene_from_config,
    load_scoring_yaml,
    load_tags,
    load_validation_status,
)
from v3_conversion.aic_meta.task_string import build_task_string
from v3_conversion.aic_meta.writer import (
    write_pose_commands_parquet,
    write_scene_parquet,
    write_scoring_parquet,
    write_task_parquet,
    write_tf_snapshots_parquet,
)
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


@dataclass(frozen=True)
class EpisodeContext:
    episode_dir: Path
    trial_dir: Optional[Path]
    trial_key: str
    trial_score_folder: str


def _load_existing_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    """Read existing parquet rows so merge mode appends instead of overwriting."""
    if not path.is_file():
        return []
    return pq.read_table(path).to_pylist()


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
        "wrench_topic": config.get("wrench_topic", ""),
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
        "wrench_topic": metacard.get("wrench_topic", defaults.get("wrench_topic", "")),
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


def _find_episode_context(run_dir: Path) -> Optional[EpisodeContext]:
    """Return episode metadata context, preferring trial metadata over root."""
    for trial_dir in sorted(run_dir.glob("trial_*")):
        episode_dir = trial_dir / "episode"
        if (episode_dir / "metadata.json").is_file():
            return EpisodeContext(
                episode_dir=episode_dir,
                trial_dir=trial_dir,
                trial_key=trial_dir.name.split("_score")[0],
                trial_score_folder=trial_dir.name,
            )

    root_episode = run_dir / "episode"
    if (root_episode / "metadata.json").is_file():
        return EpisodeContext(
            episode_dir=root_episode,
            trial_dir=None,
            trial_key="",
            trial_score_folder="",
        )

    return None


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


def _image_dir_for_camera(camera_name: str) -> str:
    lowered = camera_name.lower()
    if "left" in lowered:
        return "left"
    if "center" in lowered:
        return "center"
    if "right" in lowered:
        return "right"
    return camera_name


def _remap_missing_cameras_to_compressed(
    config,
    mcap_topics: list[str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Compute a topic_map override that swaps missing raw camera topics for
    their CompressedImage variants.

    When a bag was recorded with image_transport's compressed-only output,
    raw `/cam/image` is absent but `/cam/image/compressed` exists. This
    helper is pure: it does not touch `config`. The caller should rebuild
    the Rosbag (e.g. dataclasses.replace) with the returned topic_map when
    any remappings are reported.
    """
    available = set(mcap_topics)
    camera_canonicals = set(config.camera_names)
    new_topic_map = dict(config.topic_map)
    remapped: list[tuple[str, str]] = []

    for raw_topic in list(new_topic_map.keys()):
        canonical = new_topic_map[raw_topic]
        if canonical not in camera_canonicals:
            continue
        if raw_topic in available:
            continue
        compressed_topic = f"{raw_topic}/compressed"
        if compressed_topic not in available:
            continue
        del new_topic_map[raw_topic]
        new_topic_map[compressed_topic] = canonical
        remapped.append((raw_topic, compressed_topic))

    return new_topic_map, remapped


def _can_use_recorded_episode(
    missing_topics: list[str],
    config,
    episode_dir: Path,
) -> bool:
    camera_topics = set(config.topic_map.keys())
    if not missing_topics or not set(missing_topics).issubset(camera_topics):
        return False
    required = ("states.npy", "timestamps.npy")
    if any(not (episode_dir / name).is_file() for name in required):
        return False
    image_root = episode_dir / "images"
    return all(
        any((image_root / _image_dir_for_camera(cam)).glob("*.png"))
        for cam in config.camera_names
    )


def _infer_recorded_episode_fps(
    timestamps_sec: np.ndarray,
    fallback_fps: int,
) -> int:
    if len(timestamps_sec) < 2:
        return fallback_fps
    duration_sec = float(timestamps_sec[-1] - timestamps_sec[0])
    if duration_sec <= 0:
        return fallback_fps
    return max(1, int(round((len(timestamps_sec) - 1) / duration_sec)))


def _load_recorded_episode(
    *,
    episode_dir: Path,
    config,
    task_instruction: str,
) -> tuple[dict[str, Any], list[int], dict[str, list[int]], int]:
    states = np.load(episode_dir / "states.npy").astype(np.float32)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError(f"states.npy must be a 2D array with at least 2 frames: {states.shape}")

    obs_dim = len(config.joint_order.get("obs", []))
    if obs_dim <= 0 or states.shape[1] < obs_dim:
        raise ValueError(
            f"states.npy shape {states.shape} cannot provide {obs_dim} observation joints"
        )
    obs = states[:, -obs_dim:]

    timestamps_sec = np.load(episode_dir / "timestamps.npy")
    if len(timestamps_sec) != len(obs):
        raise ValueError(
            f"timestamps.npy has {len(timestamps_sec)} frames, expected {len(obs)}"
        )
    recorded_fps = _infer_recorded_episode_fps(timestamps_sec, config.fps)
    frame_timestamps_ns = [int(float(ts) * 1_000_000_000) for ts in timestamps_sec]

    image_root = episode_dir / "images"
    images: dict[str, list[np.ndarray]] = {}
    for camera_name in config.camera_names:
        image_dir = image_root / _image_dir_for_camera(camera_name)
        frames: list[np.ndarray] = []
        for image_path in sorted(image_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read image: {image_path}")
            frames.append(image)
        if len(frames) != len(obs):
            raise ValueError(
                f"{image_dir} has {len(frames)} images, expected {len(obs)}"
            )
        images[camera_name] = frames

    episode: dict[str, Any] = {
        "obs": obs,
        "images": images,
        "task": task_instruction,
    }

    action_cfg = config.joint_order.get("action", {})
    for action_name in config.action_order:
        width = len(action_cfg.get(action_name, []))
        if width <= 0:
            continue
        if width != obs_dim:
            raise ValueError(
                f"recorded episode fallback requires action width {width} to match obs width {obs_dim}"
            )
        episode[action_name] = obs.copy()

    optional_arrays = {
        "velocity": "joint_velocities.npy",
        "wrench": "wrenches.npy",
    }
    for key, filename in optional_arrays.items():
        path = episode_dir / filename
        if not path.is_file():
            continue
        arr = np.load(path).astype(np.float32)
        if len(arr) != len(obs):
            raise ValueError(f"{filename} has {len(arr)} frames, expected {len(obs)}")
        episode[key] = arr

    timestamps = {topic: frame_timestamps_ns for topic in config.topic_map.values()}
    return episode, frame_timestamps_ns, timestamps, recorded_fps


# ------------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------------

def run_conversion(
    config_path: str,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    merge: bool = False,
) -> int:
    """Run the full conversion pipeline.

    Parameters
    ----------
    input_dir : str, optional
        Override INPUT_PATH (where bag folders live).
    output_dir : str, optional
        Override OUTPUT_PATH.  When *merge* is True this must point to an
        existing LeRobot dataset directory.
    merge : bool
        When True, use *output_dir* as-is (no task_name subdirectory) and
        append episodes to the existing dataset found there.

    Returns exit code: 0=success, 1=all failed, 2=partial success.
    """
    global INPUT_PATH
    if input_dir is not None:
        INPUT_PATH = Path(input_dir)

    effective_output = Path(output_dir) if output_dir else OUTPUT_PATH

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
        "wrench_topic": cfg.get("wrench_topic", ""),
        "action_topics_map": cfg.get("action_topics_map", {}),
        "task_instruction": cfg.get("task_instruction", []),
        "tags": cfg.get("tags", []),
    }

    if not folders:
        logger.error("No folders to convert")
        return 1

    next_episode_index = 0
    if merge:
        output_root = str(effective_output)
        meta_info = effective_output / "meta" / "info.json"
        if not meta_info.is_file():
            logger.error(
                "Merge target is not an existing LeRobot dataset: %s "
                "(meta/info.json not found)", effective_output,
            )
            return 1

        with open(meta_info, "r", encoding="utf-8") as _f:
            existing_info = json.load(_f)
        existing_features = existing_info.get("features", {})
        existing_obs_shape = existing_features.get("observation.state", {}).get("shape", [])
        config_obs_dim = len(config_defaults["joint_names"])

        if existing_obs_shape and existing_obs_shape[0] != config_obs_dim:
            logger.error(
                "Feature mismatch: existing dataset has observation.state shape %s "
                "but config has %d joints (%s). "
                "Adjust joint_names in config to match.",
                existing_obs_shape, config_obs_dim, config_defaults["joint_names"],
            )
            return 1

        next_episode_index = int(existing_info.get("total_episodes", 0))
        logger.info(
            "MERGE mode: appending to existing dataset at %s "
            "(existing episodes=%s, fps=%s)",
            output_root,
            existing_info.get("total_episodes", "?"),
            existing_info.get("fps", "?"),
        )
    else:
        output_root = str(effective_output / task_name)

    aic_dir = Path(output_root) / "meta" / "aic"

    logger.info("Starting conversion: task=%s, folders=%d", task_name, len(folders))
    creator: Optional[DataCreator] = None
    converted_count = 0
    failed_count = 0
    skipped_count = 0
    failed_folders: List[str] = []
    skipped_reasons: List[str] = []
    if merge:
        aic_task_rows = _load_existing_parquet_rows(aic_dir / "task.parquet")
        aic_scoring_rows = _load_existing_parquet_rows(aic_dir / "scoring.parquet")
        aic_scene_rows = _load_existing_parquet_rows(aic_dir / "scene.parquet")
        aic_tf_rows = _load_existing_parquet_rows(aic_dir / "tf_snapshots.parquet")
        aic_pose_command_rows = _load_existing_parquet_rows(
            aic_dir / "pose_commands.parquet"
        )
    else:
        aic_task_rows = []
        aic_scoring_rows = []
        aic_scene_rows = []
        aic_tf_rows = []
        aic_pose_command_rows = []

    for idx, folder_name in enumerate(folders):
        logger.info("[%d/%d] Converting: %s", idx + 1, len(folders), folder_name)

        try:
            run_dir = INPUT_PATH / folder_name
            validation_status = load_validation_status(run_dir)
            if not validation_status.get("passed", False):
                skipped_count += 1
                reason = str(validation_status.get("reason", "validation failed"))
                skipped_reasons.append(f"{folder_name}: {reason}")
                logger.info("  Skipped %s: %s", folder_name, reason)
                continue

            episode_context = _find_episode_context(run_dir)
            if episode_context is None:
                skipped_count += 1
                reason = "episode/metadata.json missing"
                skipped_reasons.append(f"{folder_name}: {reason}")
                logger.info("  Skipped %s: %s", folder_name, reason)
                continue

            # 1. Load metacard (falls back to config defaults)
            metadata = _load_metacard(folder_name, config_defaults)

            # 2. Validate and build config
            mcap_path, config = _prepare_config(
                folder_name, metadata, robot_type, fps_override
            )

            # 3. Pre-check: all expected topics exist in MCAP
            validation = validate_mcap_topics(str(mcap_path), config.topic_map)
            new_topic_map, remapped_cameras = _remap_missing_cameras_to_compressed(
                config, validation.get("mcap_topics", [])
            )
            if remapped_cameras:
                for raw_t, comp_t in remapped_cameras:
                    logger.info(
                        "  Remapped missing camera topic %s -> %s "
                        "(CompressedImage fallback)",
                        raw_t, comp_t,
                    )
                config = dataclass_replace(config, topic_map=new_topic_map)
                validation = validate_mcap_topics(str(mcap_path), config.topic_map)
            use_recorded_episode = _can_use_recorded_episode(
                validation["missing_topics"],
                config,
                episode_context.episode_dir,
            )
            if validation["missing_topics"] and not use_recorded_episode:
                skipped_count += 1
                reason = (
                    "missing required MCAP topics: "
                    f"{validation['missing_topics']}"
                )
                skipped_reasons.append(f"{folder_name}: {reason}")
                logger.info("  Skipped %s: %s", folder_name, reason)
                continue

            episode_meta = load_episode_metadata(episode_context.episode_dir)
            task_instruction = build_task_string(episode_meta)
            effective_fps = config.fps

            # 5. Extract frames, or reuse collector-recorded episode artifacts
            # when the MCAP intentionally excludes camera image topics.
            if use_recorded_episode:
                logger.info(
                    "  Using recorded episode artifacts because MCAP is missing camera topics: %s",
                    validation["missing_topics"],
                )
                episode, frame_timestamps_ns, timestamps, effective_fps = _load_recorded_episode(
                    episode_dir=episode_context.episode_dir,
                    config=config,
                    task_instruction=task_instruction,
                )
                if effective_fps != config.fps:
                    logger.warning(
                        "  Recorded episode effective fps=%d differs from config fps=%d; "
                        "using effective fps to preserve camera duration.",
                        effective_fps,
                        config.fps,
                    )
                frames = []
            else:
                frames, timestamps = extract_frames(
                    bag_path=str(mcap_path), config=config,
                )

            # 6. Initialize DataCreator once the effective FPS is known.
            if creator is None:
                creator = DataCreator(
                    repo_id=repo_id,
                    root=output_root,
                    robot_type=config.robot_type,
                    action_order=config.action_order,
                    joint_order=config.joint_order,
                    camera_names=config.camera_names,
                    fps=effective_fps,
                )
            elif creator.fps != effective_fps:
                raise ValueError(
                    f"folder {folder_name} has effective fps {effective_fps}, "
                    f"but dataset was initialized with fps {creator.fps}"
                )

            # 7. Hz validation. Recorded episode artifacts may be emitted at the
            # collector's effective sampling rate rather than the converter
            # config FPS, and were already validated by the collector.
            if use_recorded_episode:
                logger.info("  [Hz] skipped for recorded episode artifacts: %s", folder_name)
            else:
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

            if not use_recorded_episode and not frames:
                raise ValueError(
                    f"No frames extracted [folder={folder_name}] from {mcap_path}. "
                    f"All expected topics present but build_frame() returned None "
                    f"for every cycle — likely cause: incomplete action data "
                    f"(missing leader topics or action key mismatch)."
                )

            if not use_recorded_episode:
                frame_timestamps_ns = [
                    int(frame["emitted_timestamp_ns"]) for frame in frames
                ]

            # 8. Transform to episode with task derived from episode metadata
            trial_dir = episode_context.trial_dir or run_dir
            trial_key = episode_context.trial_key
            run_meta = load_run_meta(run_dir)
            tags_meta = load_tags(trial_dir)
            scoring_meta = load_scoring_yaml(trial_dir, trial_key=trial_key)
            scene_meta = load_scene_from_config(run_dir, trial_key=trial_key)
            episode_start_ns = frame_timestamps_ns[0]
            pose_labels = extract_pose_labels(
                bag_path=mcap_path,
                frame_timestamps_ns=frame_timestamps_ns,
                episode_meta=episode_meta,
                base_frame="base_link",
            )
            insertion_meta = extract_insertion_event(
                mcap_path, episode_start_ns=episode_start_ns,
            )
            tf_snapshots = extract_scoring_tf_snapshots(mcap_path)
            ep_idx = next_episode_index
            pose_command_rows = extract_pose_commands(
                bag_path=mcap_path,
                episode_index=ep_idx,
                episode_start_ns=episode_start_ns,
            )
            task_row = {
                "episode_index": ep_idx,
                "run_folder": folder_name,
                "trial_key": trial_key,
                "trial_score_folder": episode_context.trial_score_folder,
                "schema_version": tags_meta["schema_version"],
                "cable_type": episode_meta["cable_type"],
                "cable_name": episode_meta["cable_name"],
                "plug_type": episode_meta["plug_type"],
                "plug_name": episode_meta["plug_name"],
                "port_type": episode_meta["port_type"],
                "port_name": episode_meta["port_name"],
                "target_module": episode_meta["target_module"],
                "success": bool(episode_meta["success"]),
                "early_terminated": bool(episode_meta["early_terminated"]),
                "early_term_source": episode_meta["early_term_source"],
                "duration_sec": float(episode_meta["duration_sec"]),
                "num_steps": int(episode_meta["num_steps"]),
                "policy": run_meta["policy"],
                "seed": int(run_meta["seed"]),
                **insertion_meta,
            }
            scoring_row = {"episode_index": ep_idx, **scoring_meta}
            scene_row = {
                "episode_index": ep_idx,
                "plug_port_distance_init": float(
                    episode_meta["plug_port_distance_init"]
                ),
                "initial_plug_pose_rel_gripper":
                    scene_meta["initial_plug_pose_rel_gripper"],
                "scene_rails": scene_meta["scene_rails"],
            }
            tf_row = {
                "episode_index": ep_idx,
                **tf_snapshots,
            }
            del timestamps

            if not use_recorded_episode:
                # frames_to_episode consumes and clears the frames list
                episode = frames_to_episode(
                    frames=frames,
                    action_order=config.action_order,
                    camera_names=config.camera_names,
                    task=task_instruction,
                )
                del frames
            episode.update(pose_labels)
            episode = apply_one_step_shift(episode)

            # 9. Convert episode
            creator.convert_episode(episode)
            del episode
            aic_task_rows.append(task_row)
            aic_scoring_rows.append(scoring_row)
            aic_scene_rows.append(scene_row)
            aic_tf_rows.append(tf_row)
            aic_pose_command_rows.extend(pose_command_rows)
            next_episode_index += 1

            converted_count += 1
            logger.info("  Converted successfully: %s", folder_name)
            gc.collect()

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
    finalize_failed = False
    if creator is not None and creator.dataset is not None:
        finalize_ok = False
        try:
            creator.dataset.finalize()
            logger.info("Dataset finalized")
            creator.correct_video_timestamps()
            logger.info("Video timestamps corrected")
            creator.patch_episodes_metadata()
            logger.info("Episode custom metadata patched")
            write_task_parquet(aic_dir / "task.parquet", aic_task_rows)
            write_scoring_parquet(aic_dir / "scoring.parquet", aic_scoring_rows)
            write_scene_parquet(aic_dir / "scene.parquet", aic_scene_rows)
            write_tf_snapshots_parquet(aic_dir / "tf_snapshots.parquet", aic_tf_rows)
            write_pose_commands_parquet(
                aic_dir / "pose_commands.parquet", aic_pose_command_rows
            )
            finalize_ok = True
        except Exception as e:
            finalize_failed = True
            logger.error("Failed to finalize dataset or write AIC metadata: %s", e)

        # Push to HuggingFace Hub if repo_id has namespace format
        if not finalize_ok:
            logger.warning(
                "Skipping push_to_hub because dataset finalization or AIC metadata write failed."
            )
        elif "/" in repo_id:
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
        "Conversion complete: %d converted, %d failed, %d skipped",
        converted_count,
        failed_count,
        skipped_count,
    )
    if failed_folders:
        logger.info("Failed folders: %s", failed_folders)
    if skipped_reasons:
        logger.info("Skipped folders: %s", skipped_reasons)

    if converted_count == 0:
        return 1
    if failed_count > 0 or finalize_failed:
        return 2
    return 0


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert MCAP ROS2 bags to LeRobot v3 dataset",
    )
    parser.add_argument(
        "config", nargs="?", default=str(DEFAULT_CONFIG),
        help="Path to config.json (default: src/config.json)",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing bag folders (overrides INPUT_PATH env/default)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. In merge mode, must be an existing LeRobot dataset.",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Append episodes to the existing dataset at --output-dir "
             "(no task_name subdirectory created)",
    )
    args = parser.parse_args()

    exit_code = run_conversion(
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        merge=args.merge,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
