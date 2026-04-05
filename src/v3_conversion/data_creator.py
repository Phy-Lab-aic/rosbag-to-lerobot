"""LeRobot v3.0 dataset creator — pure Python (lerobot + av + pyarrow).

Copied from dataset_manager.conversion.data_creator with zero changes
to the logic. This file has no ROS2 dependencies.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION as version
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


class DataCreator:
    """
    Creates LeRobot v3.0 datasets from episode data.

    Supports dynamic camera configurations - works with any number of cameras
    specified in the robot config YAML.
    """

    def __init__(
        self,
        repo_id: str,
        action_order: List[str],
        joint_order: Dict[str, Any],
        camera_names: List[str],
        fps: int = 30,
        root: str = "./datasets",
        robot_type: str = "custom_robot",
        use_videos: bool = True,
        extra_obs_config: dict = None,
        has_episode_signals: bool = False,
    ):

        self.repo_id = repo_id
        self.fps = fps
        self.root = root
        self.robot_type = robot_type
        self.use_videos = use_videos

        self.action_order = action_order
        self.joint_order = joint_order
        self.camera_names = camera_names
        self.dataset = None
        self._episode_custom_metadata: List[Dict[str, Any]] = []

        self.extra_obs_config = extra_obs_config
        self.has_episode_signals = has_episode_signals

        if "v3" not in version:
            raise RuntimeError(
                f"Unsupported LeRobot CODEBASE_VERSION={version}. "
                "DataCreator only supports LeRobot v3.0."
            )

    def load_dataset(self) -> None:
        """Load an existing LeRobot dataset for appending episodes."""
        dataset_root = Path(self.root)
        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=str(dataset_root),
        )
        self.dataset.start_image_writer(
            num_processes=0,
            num_threads=4,
        )
        self.dataset.episode_buffer = self.dataset.create_episode_buffer()

    def create_dataset(self, episode: Dict[str, Any]) -> None:

        # Infer obs_dim from actual data (may include velocity+effort)
        if "obs" in episode and len(episode["obs"]) > 0:
            obs_dim = episode["obs"].shape[-1] if hasattr(episode["obs"], 'shape') else len(episode["obs"][0])
        else:
            obs_dim = len(self.joint_order["obs"])
        action_cfg = self.joint_order["action"]
        action_joint_names = []
        for key in self.action_order:
            action_joint_names += action_cfg.get(key, [])
        action_dim = len(action_joint_names)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (obs_dim,),
                "names": self.joint_order["obs"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_joint_names,
            },
        }

        for cam_name in self.camera_names:
            if cam_name in episode["images"] and len(episode["images"][cam_name]) > 0:
                cam_img = episode["images"][cam_name][0]
                h, w, c = cam_img.shape
                features[f"observation.images.{cam_name}"] = {
                    "dtype": "video",
                    "shape": (h, w, c),
                    "names": ["height", "width", "channels"],
                }

        # Extra observation features (from extra_obs_config)
        if self.extra_obs_config:
            for name, spec in self.extra_obs_config.items():
                features[f"observation.{name}"] = {
                    "dtype": "float32",
                    "shape": spec["shape"],
                    "names": spec.get("names"),
                }

        # Done/success (episode boundary signals)
        if self.has_episode_signals:
            features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
            features["next.success"] = {"dtype": "bool", "shape": (1,), "names": None}

        dataset_root = Path(self.root)
        if dataset_root.exists():
            # Safety: only remove if it looks like a LeRobot dataset or is empty
            meta_info = dataset_root / "meta" / "info.json"
            is_lerobot_dataset = meta_info.is_file()
            is_empty = not any(dataset_root.iterdir())
            if is_lerobot_dataset or is_empty:
                logger.warning(
                    "Removing existing dataset directory: %s", dataset_root
                )
                shutil.rmtree(dataset_root)
            else:
                raise RuntimeError(
                    f"Directory exists but is not a LeRobot dataset: {dataset_root}. "
                    "Remove it manually or choose a different output path."
                )
        dataset_root.parent.mkdir(parents=True, exist_ok=True)

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=str(dataset_root),
            fps=self.fps,
            features=features,
            robot_type=self.robot_type,
            use_videos=self.use_videos,
            image_writer_processes=0,
            image_writer_threads=4,
            batch_encoding_size=1,
        )

    def convert_episode(self, episode: Dict[str, Any], custom_metadata: dict | None = None) -> None:

        if not self.dataset:
            dataset_root = Path(self.root)
            if (dataset_root / "meta" / "info.json").is_file():
                self.load_dataset()
            else:
                self.create_dataset(episode)

        if self.dataset.episode_buffer is None or "size" not in self.dataset.episode_buffer:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()

        obs = np.asarray(episode["obs"], dtype=np.float32)

        actions = []
        for key in self.action_order:
            actions.append(np.asarray(episode[key], dtype=np.float32))
        actions = np.concatenate(actions, axis=-1).astype(np.float32)

        camera_lists = {}
        for cam_name in self.camera_names:
            if cam_name in episode["images"]:
                camera_lists[cam_name] = episode["images"][cam_name]

        frame_count = obs.shape[0]
        for cam_name, cam_list in camera_lists.items():
            if len(cam_list) != frame_count:
                raise ValueError(
                    f"Camera {cam_name} has {len(cam_list)} frames, expected {frame_count}"
                )

        task = episode.get("task", "no_task_specified")
        for t in range(frame_count):
            frame = {
                "observation.state": obs[t],
                "action": actions[t],
            }
            for cam_name in self.camera_names:
                if cam_name in camera_lists:
                    frame[f"observation.images.{cam_name}"] = camera_lists[cam_name][t]

            # Extra observations
            if "extra_obs" in episode:
                for key, arr in episode["extra_obs"].items():
                    if isinstance(arr, np.ndarray) and len(arr) == frame_count:
                        frame[f"observation.{key}"] = arr[t]

            # Done/success
            if self.has_episode_signals:
                frame["next.done"] = (t == frame_count - 1)
                frame["next.success"] = (t == frame_count - 1) and episode.get("success", False)

            frame["task"] = task
            self.dataset.add_frame(frame)

        self.dataset.save_episode()

        if custom_metadata:
            self._episode_custom_metadata.append(custom_metadata)
        else:
            self._episode_custom_metadata.append({})

    def recover_dataset_state(self) -> None:
        """Close writers and reload dataset from disk after failed episode.

        Uses LeRobot internal APIs (image_writer, _close_writer) that may
        change across versions.  Pin your lerobot dependency accordingly.
        """
        if not self.dataset:
            return
        try:
            if hasattr(self.dataset, "image_writer") and self.dataset.image_writer is not None:
                self.dataset.image_writer.stop()
        except Exception as e:
            logger.debug("Failed to stop image writer during recovery: %s", e)
        try:
            self.dataset._close_writer()
        except Exception:
            pass
        try:
            self.dataset.meta._close_writer()
        except Exception:
            pass
        dataset_root = Path(self.root)
        if not (dataset_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"Dataset metadata missing, cannot recover: {dataset_root}"
            )
        self.load_dataset()

    def correct_video_timestamps(self) -> None:
        """Correct parquet timestamps to match actual video frame PTS.

        LeRobot stores parquet timestamps as frame_index/fps (float division)
        and video from_timestamp from get_video_duration_in_s(). When episodes
        are concatenated into a single mp4, rounding differences between these
        two sources accumulate, causing timestamp tolerance violations during
        training (the queried timestamp doesn't match any actual video frame).

        This method reads actual frame PTS from the encoded videos and rewrites
        the data parquet timestamp column so that (from_timestamp + timestamp)
        exactly matches each frame's PTS in the concatenated video.

        Must be called after finalize() and before patch_episodes_metadata().
        """
        if not self.dataset or not self.dataset.meta.video_keys:
            return

        meta = self.dataset.meta
        root = Path(self.dataset.root)

        from lerobot.datasets.io_utils import load_episodes
        episodes = load_episodes(root)

        primary_vid_key = meta.video_keys[0]

        # Group episodes by video file (using primary camera)
        video_groups: Dict[tuple, list] = {}
        for ep_idx in range(len(episodes)):
            ep = episodes[ep_idx]
            chunk_idx = ep[f"videos/{primary_vid_key}/chunk_index"]
            file_idx = ep[f"videos/{primary_vid_key}/file_index"]
            vf_key = (chunk_idx, file_idx)
            if vf_key not in video_groups:
                video_groups[vf_key] = []
            video_groups[vf_key].append({
                "episode_index": ep["episode_index"],
                "from_ts": ep[f"videos/{primary_vid_key}/from_timestamp"],
                "to_ts": ep[f"videos/{primary_vid_key}/to_timestamp"],
            })

        # Read actual PTS from each video file and compute per-episode corrections
        corrections: Dict[int, List[float]] = {}
        for (chunk_idx, file_idx), ep_infos in video_groups.items():
            video_path = root / meta.video_path.format(
                video_key=primary_vid_key, chunk_index=chunk_idx, file_index=file_idx,
            )
            if not video_path.is_file():
                logger.warning("Video file not found, skipping correction: %s", video_path)
                continue

            all_pts: List[float] = []
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    all_pts.append(float(frame.pts * frame.time_base))

            for info in ep_infos:
                from_ts = info["from_ts"]
                to_ts = info["to_ts"]
                ep_idx = info["episode_index"]

                ep_pts = [
                    pts for pts in all_pts
                    if from_ts - 1e-6 <= pts < to_ts + 1e-6
                ]
                ep_pts.sort()
                corrections[ep_idx] = [pts - from_ts for pts in ep_pts]

        if not corrections:
            return

        # Verify PTS consistency across all cameras (cache decoded PTS per file)
        _pts_cache: Dict[str, List[float]] = {}
        for other_vid_key in meta.video_keys[1:]:
            for (chunk_idx, file_idx), ep_infos in video_groups.items():
                other_path = root / meta.video_path.format(
                    video_key=other_vid_key, chunk_index=chunk_idx, file_index=file_idx,
                )
                if not other_path.is_file():
                    continue

                cache_key = str(other_path)
                if cache_key in _pts_cache:
                    other_pts = _pts_cache[cache_key]
                else:
                    other_pts = []
                    with av.open(str(other_path)) as container:
                        stream = container.streams.video[0]
                        for frame in container.decode(stream):
                            other_pts.append(float(frame.pts * frame.time_base))
                    _pts_cache[cache_key] = other_pts

                for info in ep_infos:
                    ep_idx = info["episode_index"]
                    if ep_idx not in corrections:
                        continue
                    from_ts = info["from_ts"]
                    to_ts = info["to_ts"]
                    other_ep_pts = sorted([
                        pts - from_ts for pts in other_pts
                        if from_ts - 1e-6 <= pts < to_ts + 1e-6
                    ])
                    primary_pts = corrections[ep_idx]
                    if len(other_ep_pts) != len(primary_pts):
                        logger.warning(
                            "PTS frame count mismatch: %s has %d frames, %s has %d "
                            "(episode %d). Timestamps corrected using %s only.",
                            primary_vid_key, len(primary_pts),
                            other_vid_key, len(other_ep_pts), ep_idx,
                            primary_vid_key,
                        )
                    elif any(
                        abs(a - b) > 1e-3
                        for a, b in zip(primary_pts, other_ep_pts)
                    ):
                        logger.warning(
                            "PTS drift detected between %s and %s "
                            "(episode %d, >1ms). Timestamps corrected using %s.",
                            primary_vid_key, other_vid_key, ep_idx,
                            primary_vid_key,
                        )

        # Group episodes by data parquet file
        data_groups: Dict[tuple, List[int]] = {}
        for ep_idx in range(len(episodes)):
            ep = episodes[ep_idx]
            chunk_idx = ep["data/chunk_index"]
            file_idx = ep["data/file_index"]
            df_key = (chunk_idx, file_idx)
            if df_key not in data_groups:
                data_groups[df_key] = []
            data_groups[df_key].append(ep["episode_index"])

        # Patch each data parquet file
        patched_count = 0
        for (chunk_idx, file_idx), ep_indices in data_groups.items():
            parquet_path = root / meta.data_path.format(
                chunk_index=chunk_idx, file_index=file_idx,
            )
            if not parquet_path.is_file():
                continue

            table = pq.read_table(parquet_path)
            ts_array = table.column("timestamp").to_pylist()
            ep_idx_array = table.column("episode_index").to_pylist()

            ep_frame_counters: Dict[int, int] = {}
            corrected = list(ts_array)
            needs_update = False

            for i, (orig_ts, ep_idx) in enumerate(zip(ts_array, ep_idx_array)):
                if ep_idx not in corrections:
                    continue
                if ep_idx not in ep_frame_counters:
                    ep_frame_counters[ep_idx] = 0
                frame_in_ep = ep_frame_counters[ep_idx]
                ep_frame_counters[ep_idx] += 1

                if frame_in_ep < len(corrections[ep_idx]):
                    new_ts = corrections[ep_idx][frame_in_ep]
                    if abs(new_ts - orig_ts) > 1e-7:
                        corrected[i] = new_ts
                        needs_update = True

            if needs_update:
                ts_col_idx = table.column_names.index("timestamp")
                new_col = pa.array(corrected, type=pa.float32())
                table = table.set_column(ts_col_idx, "timestamp", new_col)
                pq.write_table(table, parquet_path)
                patched_count += 1

        if patched_count > 0:
            logger.info(
                "Corrected video timestamps in %d parquet file(s) "
                "(%d episodes)", patched_count, len(corrections),
            )

    def patch_episodes_metadata(self) -> None:
        """Patch episodes parquet files with custom metadata after finalize()."""
        if not self._episode_custom_metadata or not self.dataset:
            return

        episodes_dir = Path(self.dataset.root) / "meta" / "episodes"
        if not episodes_dir.exists():
            return

        parquet_files = sorted(episodes_dir.rglob("*.parquet"))
        if not parquet_files:
            return

        row_offset = 0
        for pq_path in parquet_files:
            table = pq.read_table(pq_path)
            num_rows = table.num_rows

            metadata_slice = self._episode_custom_metadata[
                row_offset : row_offset + num_rows
            ]
            row_offset += num_rows

            if not metadata_slice:
                continue

            all_keys = set()
            for m in metadata_slice:
                all_keys.update(m.keys())

            for key in all_keys:
                values = [m.get(key, None) for m in metadata_slice]
                if any(isinstance(v, list) for v in values):
                    arrow_values = [v if v is not None else [] for v in values]
                    col = pa.array(arrow_values, type=pa.list_(pa.string()))
                elif all(isinstance(v, bool) for v in values if v is not None):
                    col = pa.array(values, type=pa.bool_())
                elif all(isinstance(v, (int, float)) for v in values if v is not None):
                    col = pa.array(values, type=pa.float64())
                else:
                    col = pa.array(values, type=pa.string())

                if key in table.column_names:
                    idx = table.column_names.index(key)
                    table = table.set_column(idx, key, col)
                else:
                    table = table.append_column(key, col)

            pq.write_table(table, pq_path)
