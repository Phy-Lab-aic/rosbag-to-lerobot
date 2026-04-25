"""Per-frame AIC pose label extraction from MCAP auxiliary topics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory


POSE_LABEL_KEYS = (
    "label.tcp_pose",
    "label.plug_pose_base",
    "label.port_pose_base",
    "label.target_module_pose_base",
)

_TCP_FRAME_CANDIDATES = {"tcp_link", "tool0", "tool_link"}


def _nan_pose(count: int) -> np.ndarray:
    return np.full((count, 7), np.nan, dtype=np.float32)


def _pose_from_pose_msg(pose: Any) -> np.ndarray:
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


def _pose_from_transform(transform: Any) -> np.ndarray:
    return np.array(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ],
        dtype=np.float32,
    )


def frame_id_candidates(
    name: str, cable_name: str = "", target_module: str = ""
) -> List[str]:
    """Return likely TF child-frame IDs for an AIC semantic object name."""
    base = str(name or "").strip().strip("/")
    cable = str(cable_name or "").strip().strip("/")
    module = str(target_module or "").strip().strip("/")

    candidates: list[str] = []
    if base:
        candidates.extend([base, f"{base}_link", f"{base}_link_entrance"])
    if cable and base:
        candidates.extend(
            [
                f"{cable}/{base}",
                f"{cable}/{base}_link",
                f"{cable}/{base}_link_entrance",
            ]
        )
    if module and base:
        candidates.extend(
            [
                f"task_board/{module}/{base}",
                f"task_board/{module}/{base}_link",
                f"task_board/{module}/{base}_link_entrance",
            ]
        )
    if module and base == module:
        candidates.extend(
            [
                f"task_board/{module}",
                f"task_board/{module}/{module}_link",
            ]
        )
    return list(dict.fromkeys(candidates))


def _decoded_messages(bag_path: Path, wanted_topics: set[str]):
    decoder_factory = DecoderFactory()
    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    decoders: dict[int, Any] = {}

    try:
        with Path(bag_path).open("rb") as f:
            for record in StreamReader(f, record_size_limit=None).records:
                record_type = type(record).__name__
                if record_type == "Schema":
                    schemas[record.id] = record
                elif record_type == "Channel":
                    channels[record.id] = record
                elif record_type == "Message":
                    channel = channels.get(record.channel_id)
                    if channel is None or channel.topic not in wanted_topics:
                        continue
                    schema = schemas.get(channel.schema_id)
                    if schema is None:
                        continue
                    if record.channel_id not in decoders:
                        decoders[record.channel_id] = decoder_factory.decoder_for(
                            channel.message_encoding, schema
                        )
                    decoder = decoders[record.channel_id]
                    if decoder is None:
                        continue
                    try:
                        yield channel.topic, int(record.log_time), decoder(record.data)
                    except Exception:
                        continue
    except Exception:
        return


def _sample_previous(
    samples: list[tuple[int, np.ndarray]], t_ns: int
) -> tuple[np.ndarray | None, bool]:
    selected = None
    for sample_t, pose in samples:
        if sample_t <= t_ns:
            selected = pose
        else:
            break
    if selected is None:
        return None, False
    return selected, True


def _fill_from_samples(
    target: np.ndarray,
    valid: np.ndarray,
    samples: list[tuple[int, np.ndarray]],
    frame_timestamps_ns: Iterable[int],
) -> None:
    samples.sort(key=lambda item: item[0])
    for idx, t_ns in enumerate(frame_timestamps_ns):
        pose, ok = _sample_previous(samples, int(t_ns))
        if ok:
            target[idx] = pose
            valid[idx] = True


def _new_result(count: int) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {key: _nan_pose(count) for key in POSE_LABEL_KEYS}
    result.update(
        {f"{key}_valid": np.zeros((count,), dtype=np.bool_) for key in POSE_LABEL_KEYS}
    )
    return result


def _child_frame_id(transform_stamped: Any) -> str:
    return str(getattr(transform_stamped, "child_frame_id", "")).strip("/")


def _parent_frame_id(transform_stamped: Any) -> str:
    header = getattr(transform_stamped, "header", None)
    return str(getattr(header, "frame_id", "")).strip("/")


def extract_pose_labels(
    bag_path: Path,
    frame_timestamps_ns: list[int],
    episode_meta: Dict[str, Any],
    base_frame: str = "base_link",
) -> Dict[str, np.ndarray]:
    """Extract per-frame AIC pose labels from controller state and TF topics."""
    frame_times = [int(t_ns) for t_ns in frame_timestamps_ns]
    result = _new_result(len(frame_times))
    normalized_base_frame = str(base_frame or "").strip("/")

    tcp_samples: list[tuple[int, np.ndarray]] = []
    tf_tcp_samples: list[tuple[int, np.ndarray]] = []
    scoring_samples: dict[str, list[tuple[int, np.ndarray]]] = {
        "label.plug_pose_base": [],
        "label.port_pose_base": [],
        "label.target_module_pose_base": [],
    }

    cable_name = str(episode_meta.get("cable_name", ""))
    target_module = str(episode_meta.get("target_module", ""))
    plug_candidates = set(
        frame_id_candidates(
            str(episode_meta.get("plug_name", "")),
            cable_name=cable_name,
            target_module=target_module,
        )
    )
    port_candidates = set(
        frame_id_candidates(
            str(episode_meta.get("port_name", "")),
            cable_name=cable_name,
            target_module=target_module,
        )
    )
    target_candidates = set(
        frame_id_candidates(
            target_module,
            cable_name=cable_name,
            target_module=target_module,
        )
    )

    for topic, t_ns, msg in _decoded_messages(
        Path(bag_path),
        {"/aic_controller/controller_state", "/tf", "/scoring/tf"},
    ):
        if topic == "/aic_controller/controller_state" and hasattr(msg, "tcp_pose"):
            tcp_samples.append((t_ns, _pose_from_pose_msg(msg.tcp_pose)))
            continue

        transforms = getattr(msg, "transforms", [])
        if topic == "/tf":
            for transform_stamped in transforms:
                if _parent_frame_id(transform_stamped) != normalized_base_frame:
                    continue
                if _child_frame_id(transform_stamped) in _TCP_FRAME_CANDIDATES:
                    tf_tcp_samples.append(
                        (t_ns, _pose_from_transform(transform_stamped.transform))
                    )
            continue

        if topic == "/scoring/tf":
            for transform_stamped in transforms:
                if _parent_frame_id(transform_stamped) != normalized_base_frame:
                    continue
                child = _child_frame_id(transform_stamped)
                pose = _pose_from_transform(transform_stamped.transform)
                if child in plug_candidates:
                    scoring_samples["label.plug_pose_base"].append((t_ns, pose))
                if child in port_candidates:
                    scoring_samples["label.port_pose_base"].append((t_ns, pose))
                if child in target_candidates:
                    scoring_samples["label.target_module_pose_base"].append(
                        (t_ns, pose)
                    )

    tcp_source = tcp_samples if tcp_samples else tf_tcp_samples
    _fill_from_samples(
        result["label.tcp_pose"],
        result["label.tcp_pose_valid"],
        tcp_source,
        frame_times,
    )
    for key, samples in scoring_samples.items():
        _fill_from_samples(
            result[key],
            result[f"{key}_valid"],
            samples,
            frame_times,
        )

    return result
