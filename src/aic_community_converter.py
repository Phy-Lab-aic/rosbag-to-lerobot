"""
aic_community (신규/구 포맷) → LeRobot v3 변환기

지원 포맷:
  Format B (e2e): run_dir/bag/ + episode/
  Format A (raw/backup): run_dir/trial_N_*/bag/ + episode/

리샘플링 전략 (target_fps=20):
  - 이미지  : MCAP (~5Hz) → nearest-neighbor  (없으면 episode/images/ fallback)
  - state/action : episode/ → linear interpolation
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory

from v3_conversion.data_creator import DataCreator
from v3_conversion.utils import compute_grade

logger = logging.getLogger(__name__)


# ── feature 정의 ──────────────────────────────────────────────────────────────
STATE_NAMES: List[str] = [
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
    "ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z",
    "ee_vel_x", "ee_vel_y", "ee_vel_z",
    "ee_angvel_x", "ee_angvel_y", "ee_angvel_z",
    "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
    "joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper",
]

ACTION_NAMES: List[str] = [
    "target_pos_x", "target_pos_y", "target_pos_z",
    "target_quat_w", "target_quat_x", "target_quat_y", "target_quat_z",
]

# LeRobot 카메라 키 → MCAP 토픽
DEFAULT_IMAGE_TOPIC_MAP: Dict[str, str] = {
    "cam_left":   "/left_camera/image/compressed",
    "cam_center": "/center_camera/image/compressed",
    "cam_right":  "/right_camera/image/compressed",
}

# LeRobot 카메라 키 → episode/images/ 서브디렉토리 (fallback)
EPISODE_IMAGE_DIR_MAP: Dict[str, str] = {
    "cam_left":   "left",
    "cam_center": "center",
    "cam_right":  "right",
}


# ── 에피소드 탐색 ─────────────────────────────────────────────────────────────
def find_episodes(input_dir: Path, max_episodes: int = 0) -> List[Tuple[Path, Path]]:
    """episode/states.npy 가 있는 run/trial 폴더를 반환.

    Format B (e2e): run_dir/episode/states.npy
    Format A (raw/backup): run_dir/trial_N_*/episode/states.npy

    max_episodes=0 이면 전체 변환.
    """
    result = []
    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ep_dir = run_dir / "episode"
        if (ep_dir / "states.npy").is_file():
            # Format B
            result.append((run_dir, ep_dir))
        else:
            # Format A: trial 서브디렉토리 탐색
            for trial_dir in sorted(run_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial"):
                    continue
                tep_dir = trial_dir / "episode"
                if (tep_dir / "states.npy").is_file():
                    result.append((trial_dir, tep_dir))
    if max_episodes > 0:
        result = result[:max_episodes]
    return result


# ── MCAP 이미지 추출 ──────────────────────────────────────────────────────────
def _find_mcap(run_dir: Path) -> Optional[Path]:
    """run_dir/bag/*.mcap 를 찾아 반환. 없으면 None."""
    bag_dir = run_dir / "bag"
    if bag_dir.is_dir():
        mcaps = sorted(bag_dir.glob("*.mcap"))
        if mcaps:
            return mcaps[0]
    return None


def _extract_mcap_images(
    bag_path: Path,
    image_topic_map: Dict[str, str],
) -> Dict[str, Tuple[np.ndarray, List[np.ndarray]]]:
    """MCAP에서 카메라 이미지와 타임스탬프를 추출.

    Returns:
        {cam_key: (timestamps_sec ndarray, [rgb_frame, ...])}
    """
    topic_to_key = {v: k for k, v in image_topic_map.items()}
    target_topics = set(topic_to_key.keys())

    buf: Dict[str, Tuple[List[float], List[np.ndarray]]] = {
        k: ([], []) for k in image_topic_map
    }

    df = DecoderFactory()
    schemas: dict = {}
    channels: dict = {}
    decoders: dict = {}

    with open(bag_path, "rb") as f:
        try:
            for rec in StreamReader(f, record_size_limit=None).records:
                rtype = type(rec).__name__
                if rtype == "Schema":
                    schemas[rec.id] = rec
                elif rtype == "Channel":
                    channels[rec.id] = rec
                elif rtype == "Message":
                    ch = channels.get(rec.channel_id)
                    if ch is None or ch.topic not in target_topics:
                        continue
                    sc = schemas.get(ch.schema_id)
                    if sc is None:
                        continue
                    if rec.channel_id not in decoders:
                        decoders[rec.channel_id] = df.decoder_for(ch.message_encoding, sc)
                    dec = decoders[rec.channel_id]
                    if dec is None:
                        continue
                    try:
                        msg = dec(rec.data)
                    except Exception:
                        continue

                    cam_key = topic_to_key[ch.topic]
                    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    buf[cam_key][0].append(rec.log_time / 1e9)
                    buf[cam_key][1].append(frame_rgb)
        except Exception:
            pass  # 잘린 MCAP 허용

    return {
        k: (np.array(ts), frames)
        for k, (ts, frames) in buf.items()
        if ts
    }


# ── 에피소드 이미지 로딩 (fallback) ──────────────────────────────────────────
def _load_episode_images(
    episode_dir: Path,
    frame_count: int,
    camera_names: List[str],
) -> Dict[str, List[np.ndarray]]:
    """episode/images/ PNG 파일에서 이미지 로딩."""
    from PIL import Image as PILImage
    frames: Dict[str, List[np.ndarray]] = {}
    for cam_key in camera_names:
        folder = EPISODE_IMAGE_DIR_MAP.get(cam_key)
        if folder is None:
            continue
        img_dir = episode_dir / "images" / folder
        if not img_dir.is_dir():
            continue
        cam_frames = []
        for i in range(frame_count):
            p = img_dir / f"{i:04d}.png"
            if not p.is_file():
                break
            cam_frames.append(np.asarray(PILImage.open(p).convert("RGB"), dtype=np.uint8))
        if len(cam_frames) == frame_count:
            frames[cam_key] = cam_frames
    return frames


# ── 리샘플링 ─────────────────────────────────────────────────────────────────
def _resample_to_fps(
    target_fps: int,
    action_timestamps: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    cam_data: Dict[str, Tuple[np.ndarray, List[np.ndarray]]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[np.ndarray]]]:
    """state/action은 linear interpolation, 이미지는 nearest-neighbor로 리샘플링."""
    t_start = action_timestamps[0]
    t_end   = action_timestamps[-1]

    dt = 1.0 / target_fps
    t_grid = np.arange(t_start, t_end, dt)
    N = len(t_grid)

    resampled_states = np.stack(
        [np.interp(t_grid, action_timestamps, states[:, i])
         for i in range(states.shape[1])],
        axis=1,
    ).astype(np.float32)

    resampled_actions = np.stack(
        [np.interp(t_grid, action_timestamps, actions[:, i])
         for i in range(actions.shape[1])],
        axis=1,
    ).astype(np.float32)

    resampled_images: Dict[str, List[np.ndarray]] = {}
    for cam_key, (cam_ts, frames) in cam_data.items():
        indices = np.argmin(
            np.abs(cam_ts[:, None] - t_grid[None, :]), axis=0
        )
        resampled_images[cam_key] = [frames[int(i)] for i in indices]

    logger.info(
        "  리샘플링: %d steps (%.1fHz) → %d steps (%dHz)",
        len(action_timestamps),
        len(action_timestamps) / (t_end - t_start),
        N,
        target_fps,
    )
    return resampled_states, resampled_actions, resampled_images


# ── 에피소드 로딩 ─────────────────────────────────────────────────────────────
def _load_episode(
    run_dir: Path,
    episode_dir: Path,
    task_instruction: str,
    camera_names: List[str],
    image_topic_map: Dict[str, str],
    target_fps: int,
) -> Tuple[dict, dict, dict]:
    states     = np.load(episode_dir / "states.npy").astype(np.float32)
    actions    = np.load(episode_dir / "actions.npy").astype(np.float32)
    timestamps = np.load(episode_dir / "timestamps.npy").astype(np.float64)

    if states.shape[1] != len(STATE_NAMES):
        raise ValueError(f"states dim mismatch: {states.shape[1]} != {len(STATE_NAMES)}")
    if actions.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"actions dim mismatch: {actions.shape[1]} != {len(ACTION_NAMES)}")

    mcap_path = _find_mcap(run_dir)
    cam_data: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = {}

    if mcap_path is not None:
        logger.info("  MCAP에서 이미지 추출: %s", mcap_path.name)
        cam_data = _extract_mcap_images(mcap_path, {
            k: v for k, v in image_topic_map.items() if k in camera_names
        })
        missing = [k for k in camera_names if k not in cam_data]
        if missing:
            logger.warning("  MCAP에서 누락된 카메라: %s — episode/images/ fallback", missing)

    missing_cams = [k for k in camera_names if k not in cam_data]
    if missing_cams:
        fallback = _load_episode_images(episode_dir, len(timestamps), missing_cams)
        for cam_key, frames in fallback.items():
            cam_data[cam_key] = (timestamps.copy(), frames)

    if not cam_data:
        raise ValueError("카메라 이미지를 하나도 로딩하지 못했습니다")

    r_states, r_actions, r_images = _resample_to_fps(
        target_fps, timestamps, states, actions, cam_data
    )

    episode = {
        "obs":    r_states,
        "action": r_actions,
        "images": r_images,
        "task":   task_instruction,
    }

    tags: dict = {}
    for meta_path in [run_dir / "tags.json", episode_dir / "metadata.json"]:
        if meta_path.is_file():
            with open(meta_path) as f:
                tags = json.load(f)
            break

    scoring = tags.get("scoring", {})
    custom_meta = {
        "run_id":       run_dir.name,
        "success":      str(tags.get("success", "")),
        "grade":        compute_grade(tags),
        "cable_type":   str(tags.get("cable_type", "")),
        "plug_type":    str(tags.get("plug_type", "")),
        "port_type":    str(tags.get("port_type", "")),
        "score_total":  str(scoring.get("total", "") if isinstance(scoring, dict) else ""),
        "tier_3_score": str(scoring.get("tier_3_score", "") if isinstance(scoring, dict) else ""),
        "duration_sec": str(tags.get("trial_duration_sec", "")),
        "policy":       str(tags.get("policy", "")),
    }

    return episode, custom_meta, tags


# ── 변환 실행 ─────────────────────────────────────────────────────────────────
def run_conversion(config_path: str) -> int:
    with open(config_path) as f:
        cfg = json.load(f)

    raw_input_dirs  = cfg.get("input_dirs") or [cfg["input_dir"]]
    input_dirs      = [Path(d) for d in raw_input_dirs]
    output_dir      = Path(cfg["output_dir"])
    task_name       = cfg["task"]
    repo_id         = cfg.get("repo_id", task_name)
    target_fps      = int(cfg.get("fps", 20))
    robot_type      = cfg.get("robot", "ur5e")
    task_instr      = cfg.get("task_instruction", "default_task")
    cam_names       = cfg.get("camera_names", ["cam_left", "cam_center", "cam_right"])
    image_topic_map = cfg.get("image_topic_map", DEFAULT_IMAGE_TOPIC_MAP)
    max_episodes    = int(cfg.get("max_episodes", 0))
    success_only    = bool(cfg.get("success_only", False))
    push_hub        = bool(cfg.get("push_to_hub", False))

    assert len(STATE_NAMES) == 26
    assert len(ACTION_NAMES) == 7

    output_root = str(output_dir / task_name)

    all_episodes: List[Tuple[Path, Path]] = []
    for d in input_dirs:
        eps = find_episodes(d, max_episodes=0)
        logger.info("소스 %s: %d개 에피소드 발견", d, len(eps))
        all_episodes.extend(eps)

    if max_episodes > 0:
        all_episodes = all_episodes[:max_episodes]

    logger.info("총 에피소드: %d개 | target_fps=%d | success_only=%s",
                len(all_episodes), target_fps, success_only)

    creator = DataCreator(
        repo_id=repo_id,
        root=output_root,
        robot_type=robot_type,
        action_order=["action"],
        joint_order={
            "obs":    STATE_NAMES,
            "action": {"action": ACTION_NAMES},
        },
        camera_names=cam_names,
        fps=target_fps,
    )

    converted, failed, skipped = 0, 0, 0
    failed_list: List[str] = []
    total = len(all_episodes)

    for idx, (run_dir, ep_dir) in enumerate(all_episodes):
        label = run_dir.name
        pct = (idx + 1) / total * 100
        logger.info("[%d/%d %.1f%%] %s", idx + 1, total, pct, label)
        try:
            episode, custom_meta, tags = _load_episode(
                run_dir, ep_dir, task_instr, cam_names, image_topic_map, target_fps
            )

            if success_only and not tags.get("success", False):
                logger.info("  SKIP (success=False)")
                skipped += 1
                continue

            creator.convert_episode(episode, custom_metadata=custom_meta)
            converted += 1
            logger.info("  완료 (%d frames) grade=%s", episode["obs"].shape[0],
                        custom_meta.get("grade", ""))
        except Exception as e:
            failed += 1
            failed_list.append(label)
            logger.error("  실패 [%s]: %s\n%s", label, e, traceback.format_exc())
            if creator.dataset is not None:
                try:
                    creator.recover_dataset_state()
                except Exception as re:
                    logger.error("  복구 실패: %s", re)
                    creator.dataset = None

    if creator.dataset is not None:
        try:
            creator.dataset.finalize()
            creator.correct_video_timestamps()
            creator.patch_episodes_metadata()
            logger.info("데이터셋 완성 → %s", output_root)
        except Exception as e:
            logger.error("Finalize 오류: %s", e)

    logger.info("완료: 성공 %d, 스킵 %d, 실패 %d", converted, skipped, failed)
    if failed_list:
        logger.info("실패 목록: %s", failed_list)

    if push_hub and "/" in repo_id and converted > 0:
        _push_to_hub(repo_id, output_root, cfg)

    return 0 if failed == 0 else (1 if converted == 0 else 2)


def _push_to_hub(repo_id: str, output_root: str, cfg: dict) -> None:
    """HuggingFace Hub에 데이터셋 업로드."""
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        try:
            private = bool(cfg.get("private", False))
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
            logger.info("HF repo 준비 완료: %s", repo_id)
        except Exception as e:
            logger.warning("repo 생성 오류 (무시): %s", e)

        logger.info("HuggingFace Hub 업로드 시작: %s → %s", output_root, repo_id)
        api.upload_folder(
            folder_path=output_root,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="aic_community_converter full conversion",
        )
        logger.info("HuggingFace Hub 업로드 완료: https://huggingface.co/datasets/%s", repo_id)
    except Exception as e:
        logger.error("HuggingFace Hub 업로드 실패: %s\n%s", e, traceback.format_exc())
