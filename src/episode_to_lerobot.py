"""episode/ 폴더 기반 → LeRobot v3 변환기.

MCAP에 이미지 토픽이 없는 배치(backup 데이터, _050536 배치)를 위한 변환 경로.
각 trial의 episode/ 폴더에서 PNG + npy를 직접 읽어 변환한다.

episode/ 구조:
  trial_N_scoreN/
    episode/
      images/
        left/   0000.png, 0001.png, ...
        center/
        right/
      states.npy      (N, 26)  — [19:26]이 joint_positions
      actions.npy     (N, 7)
      timestamps.npy  (N,)
      metadata.json

Usage:
  python episode_to_lerobot.py
  python episode_to_lerobot.py --runs-root ~/data_backup/aic_community_e2e_backup
  python episode_to_lerobot.py --success-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from v3_conversion.data_creator import DataCreator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_hrchung.json"
DEFAULT_RUNS_ROOT = Path.home() / "aic_community_raw_data" / "runs"
DEFAULT_OUTPUT = Path.home() / "aic_community_raw_data" / "lerobot"

_EPISODE_CAM_MAP = {
    "left": "cam_left",
    "center": "cam_center",
    "right": "cam_right",
}


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
        "joint_names": cfg.get("joint_names", []),
        "action_topics_map": cfg.get("action_topics_map", {}),
        "task_instruction": cfg.get("task_instruction", []),
        "tags": cfg.get("tags", []),
    }


def compute_grade(tags: dict) -> str:
    if not tags.get("success", False):
        return "Bad"
    scoring = tags.get("scoring", {})
    tier_2 = scoring.get("total", 0) - 1 - scoring.get("tier_3_score", 0)
    return "Good" if tier_2 >= 20.0 else "Normal"


def _find_trials(runs_root: Path, success_only: bool) -> List[Dict[str, Any]]:
    trials = []
    for run_dir in sorted(runs_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        for trial_dir in sorted(run_dir.glob("trial_*")):
            ep_dir = trial_dir / "episode"
            if not ep_dir.is_dir():
                continue
            if not (ep_dir / "states.npy").is_file():
                continue

            tags: dict = {}
            tags_path = trial_dir / "tags.json"
            if tags_path.is_file():
                try:
                    tags = json.loads(tags_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning("Could not read %s: %s", tags_path, e)

            if success_only and not tags.get("success", False):
                logger.info("Skipping (not success): %s/%s", run_dir.name, trial_dir.name)
                continue

            trials.append({
                "run": run_dir.name,
                "trial": trial_dir.name,
                "episode_dir": ep_dir,
                "tags": tags,
            })
    return trials


def _load_png_sequence(img_dir: Path) -> List[np.ndarray]:
    files = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))
    frames = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            raise ValueError(f"Failed to read image: {f}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def _extract_episode(
    episode_dir: Path,
    camera_names: List[str],
    action_order: List[str],
    task: str,
) -> Dict[str, Any]:
    """episode/ 폴더에서 episode dict 구성.

    states.npy 컬럼 구조 (ControllerState):
      [0:7]   tcp_pose, [7:13] tcp_velocity, [13:19] tcp_error
      [19:26] joint_positions ← observation.state
    """
    raw_states = np.load(episode_dir / "states.npy").astype(np.float32)
    states = raw_states[:, 19:26]  # (N, 7) joint positions
    actions = np.load(episode_dir / "actions.npy").astype(np.float32)

    images: Dict[str, List[np.ndarray]] = {}
    img_root = episode_dir / "images"
    for ep_key, cam_name in _EPISODE_CAM_MAP.items():
        if cam_name not in camera_names:
            continue
        cam_dir = img_root / ep_key
        if cam_dir.is_dir():
            images[cam_name] = _load_png_sequence(cam_dir)

    n_frames = states.shape[0]
    for cam_name, imgs in images.items():
        if len(imgs) != n_frames:
            raise ValueError(f"Camera {cam_name}: {len(imgs)} frames but states has {n_frames}")

    episode: Dict[str, Any] = {"obs": states, "images": images, "task": task}
    if action_order:
        episode[action_order[0]] = actions
        for key in action_order[1:]:
            episode[key] = np.zeros((n_frames, 0), dtype=np.float32)

    return episode


def run_episode_conversion(
    runs_root: Path,
    output_root: Path,
    config_path: Path,
    success_only: bool = False,
) -> int:
    cfg = _load_config(config_path)

    task_name = cfg["task_name"]
    repo_id = cfg["repo_id"]
    fps = cfg["fps"]
    output_dir = str(output_root / task_name)
    joint_names = cfg["joint_names"]
    action_order = list(cfg.get("action_topics_map", {"leader": ""}).keys())
    camera_names = list(_EPISODE_CAM_MAP.values())

    task_instruction_list = cfg.get("task_instruction", [])
    task = task_instruction_list[0] if task_instruction_list else "default_task"

    trials = _find_trials(runs_root, success_only)
    if not trials:
        logger.error("No trials with episode/ found under %s", runs_root)
        return 1

    first_ep_dir = trials[0]["episode_dir"]
    action_dim = np.load(first_ep_dir / "actions.npy").shape[1]
    action_names = joint_names if len(joint_names) == action_dim else [f"a{i}" for i in range(action_dim)]
    joint_order = {
        "obs": joint_names,
        "action": {key: action_names for key in action_order},
    }

    logger.info(
        "Starting conversion: task=%s, trials=%d, output=%s",
        task_name, len(trials), output_dir,
    )
    logger.info("obs_dim=7 (joint_positions), action_dim=%d, cameras=%s", action_dim, camera_names)

    creator: Optional[DataCreator] = None
    converted = 0
    failed = 0

    for idx, trial_info in enumerate(trials):
        label = f"{trial_info['run']}/{trial_info['trial']}"
        logger.info("[%d/%d] Converting: %s", idx + 1, len(trials), label)

        try:
            episode = _extract_episode(trial_info["episode_dir"], camera_names, action_order, task)

            if creator is None:
                creator = DataCreator(
                    repo_id=repo_id,
                    root=output_dir,
                    robot_type=cfg["robot_type"],
                    action_order=action_order,
                    joint_order=joint_order,
                    camera_names=camera_names,
                    fps=fps,
                )

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
            logger.info("  Converted: %s", label)

        except Exception as e:
            failed += 1
            logger.error("  Failed: %s — %s\n%s", label, e, traceback.format_exc())
            if creator is not None:
                try:
                    creator.recover_dataset_state()
                except Exception as re:
                    logger.error("  Recovery failed: %s", re)
                    creator.dataset = None

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

    logger.info("Conversion complete: %d converted, %d failed", converted, failed)
    if converted == 0:
        return 1
    return 2 if failed > 0 else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()

    sys.exit(
        run_episode_conversion(
            runs_root=args.runs_root,
            output_root=args.output,
            config_path=args.config,
            success_only=args.success_only,
        )
    )


if __name__ == "__main__":
    main()
