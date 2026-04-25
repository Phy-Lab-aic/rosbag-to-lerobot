"""Per-frame AIC pose label extraction from MCAP auxiliary topics."""

from __future__ import annotations

import re
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
_NUMERIC_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_\d+$")


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


def _stripped_numeric_alias(name: str) -> str:
    match = _NUMERIC_SUFFIX_RE.match(name)
    if match is None:
        return ""
    return match.group("stem")


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
        alias = _stripped_numeric_alias(module)
        if alias:
            candidates.extend(
                [
                    alias,
                    f"{alias}_link",
                    f"task_board/{module}/{alias}",
                    f"task_board/{module}/{alias}_link",
                    f"task_board/{module}/{alias}_link_entrance",
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
                        try:
                            decoders[record.channel_id] = decoder_factory.decoder_for(
                                channel.message_encoding, schema
                            )
                        except Exception:
                            decoders[record.channel_id] = None
                            continue
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


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def _quat_normalized(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _rotate_vector(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    normalized = _quat_normalized(quat)
    vector_quat = np.array([vector[0], vector[1], vector[2], 0.0], dtype=np.float64)
    rotated = _quat_multiply(
        _quat_multiply(normalized, vector_quat), _quat_conjugate(normalized)
    )
    return rotated[:3]


def _compose_poses(parent_to_mid: np.ndarray, mid_to_child: np.ndarray) -> np.ndarray:
    first = parent_to_mid.astype(np.float64)
    second = mid_to_child.astype(np.float64)
    first_q = first[3:7]
    second_q = second[3:7]
    translation = first[:3] + _rotate_vector(first_q, second[:3])
    rotation = _quat_normalized(_quat_multiply(first_q, second_q))
    return np.array([*translation, *rotation], dtype=np.float32)


def _compose_path_to_base(
    transforms_by_child: dict[str, tuple[str, np.ndarray]],
    base_frame: str,
    child_frame: str,
) -> np.ndarray | None:
    chain: list[np.ndarray] = []
    seen: set[str] = set()
    current = child_frame
    if current == base_frame:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    while current != base_frame:
        if current in seen:
            return None
        seen.add(current)
        parent_and_pose = transforms_by_child.get(current)
        if parent_and_pose is None:
            return None
        parent, pose = parent_and_pose
        chain.append(pose)
        current = parent

    composed = chain.pop()
    while chain:
        composed = _compose_poses(composed, chain.pop())
    return composed


def _append_scoring_sample_if_available(
    scoring_samples: dict[str, list[tuple[int, np.ndarray]]],
    transforms_by_child: dict[str, tuple[str, np.ndarray]],
    label_key: str,
    child_frame: str,
    t_ns: int,
    base_frame: str,
) -> None:
    pose = _compose_path_to_base(transforms_by_child, base_frame, child_frame)
    if pose is None:
        _, pose = transforms_by_child[child_frame]
    scoring_samples[label_key].append((t_ns, pose))


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
    scoring_transforms: dict[str, tuple[str, np.ndarray]] = {}

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
                child = _child_frame_id(transform_stamped)
                if not child:
                    continue
                scoring_transforms[child] = (
                    _parent_frame_id(transform_stamped),
                    _pose_from_transform(transform_stamped.transform),
                )

            for child in plug_candidates.intersection(scoring_transforms):
                _append_scoring_sample_if_available(
                    scoring_samples,
                    scoring_transforms,
                    "label.plug_pose_base",
                    child,
                    t_ns,
                    normalized_base_frame,
                )
            for child in port_candidates.intersection(scoring_transforms):
                _append_scoring_sample_if_available(
                    scoring_samples,
                    scoring_transforms,
                    "label.port_pose_base",
                    child,
                    t_ns,
                    normalized_base_frame,
                )
            for child in target_candidates.intersection(scoring_transforms):
                _append_scoring_sample_if_available(
                    scoring_samples,
                    scoring_transforms,
                    "label.target_module_pose_base",
                    child,
                    t_ns,
                    normalized_base_frame,
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
