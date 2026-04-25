"""MCAP file I/O for the standalone conversion pipeline.

Replaces rosbag2_py with pure Python mcap + mcap-ros2-support.
All frame extraction and config assembly logic is preserved from
the original dataset_manager.conversion.mcap_reader.
"""

from typing import Any, Dict, List, Tuple

from mcap.reader import make_reader, NonSeekingReader, SeekingReader
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory

from v3_conversion.data_spec import Rosbag
from v3_conversion.data_converter import build_frame

HZ_MIN_RATIO = 0.7


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _read_rosbag_messages(bag_path: str):
    """Yield ``(topic, msg, timestamp, schema_name)`` from an MCAP bag.

    Uses a forward-only StreamReader + DecoderFactory for CDR
    deserialization.  Tolerates truncated MCAP files (missing footer)
    by catching errors at the record level.
    """
    decoder_factory = DecoderFactory()
    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    decoders: dict[int, Any] = {}  # channel_id -> decoder fn

    with open(bag_path, "rb") as f:
        try:
            for record in StreamReader(f, record_size_limit=None).records:
                rtype = type(record).__name__
                if rtype == "Schema":
                    schemas[record.id] = record
                elif rtype == "Channel":
                    channels[record.id] = record
                elif rtype == "Message":
                    ch = channels.get(record.channel_id)
                    if ch is None:
                        continue
                    sc = schemas.get(ch.schema_id)
                    if sc is None:
                        continue
                    # Lazily build decoder per channel
                    if record.channel_id not in decoders:
                        decoders[record.channel_id] = decoder_factory.decoder_for(
                            ch.message_encoding, sc
                        )
                    dec_fn = decoders[record.channel_id]
                    if dec_fn is None:
                        continue
                    try:
                        decoded_msg = dec_fn(record.data)
                    except Exception:
                        continue
                    yield ch.topic, decoded_msg, record.log_time, sc.name
        except Exception:
            return  # stop gracefully at truncated region


def _resolve_action_topics(
    action_topics_map: dict[str, str],
) -> dict[str, str]:
    """Resolve action topics and map to canonical names."""
    action_topic_to_canonical: dict[str, str] = {}
    for source_name, topic in action_topics_map.items():
        if source_name == "leader":
            canonical = "action"
        elif source_name.startswith("leader_"):
            canonical = source_name.replace("leader_", "action_", 1)
        else:
            raise ValueError(
                f"invalid action source name '{source_name}': expected 'leader' or 'leader_*'"
            )
        action_topic_to_canonical[topic] = canonical
    return action_topic_to_canonical


_SIDE_ALIASES: dict[str, list[str]] = {
    "left": ["left", "_l_"],
    "right": ["right", "_r_"],
}


def _matches_side(name: str, patterns: list[str]) -> bool:
    """Check if a joint name matches side patterns (word boundary aware)."""
    lower = name.lower()
    for p in patterns:
        if p in ("left", "right"):
            # Full word: must appear as a segment (e.g. "left_finger" but not "eleft")
            if p in lower and (
                lower.startswith(p) or lower.endswith(p)
                or f"_{p}" in lower or f"{p}_" in lower
            ):
                return True
        elif p in lower:
            return True
    return False


def _build_action_joint_order(
    action_order: list[str],
    joint_names: list[str],
) -> dict[str, list[str]]:
    action_joint_order: dict[str, list[str]] = {}
    for action_name in action_order:
        if action_name == "action":
            action_joint_order[action_name] = list(joint_names)
            continue

        suffix = action_name.replace("action_", "", 1).lower()
        if suffix:
            patterns = _SIDE_ALIASES.get(suffix, [suffix])
            action_joint_order[action_name] = [
                name for name in joint_names
                if _matches_side(name, patterns)
            ]
            continue

        action_joint_order[action_name] = list(joint_names)

    return action_joint_order


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def validate_mcap_topics(
    bag_path: str, topic_map: dict[str, str]
) -> dict[str, Any]:
    """Validate that expected topics exist in MCAP file.

    Tries summary-based read first (fast). Falls back to a forward scan
    of channel records for truncated MCAP files that lack a valid footer.
    """
    mcap_topics: list[str] = []

    # Try summary first (fast path)
    try:
        with open(bag_path, "rb") as f:
            reader = SeekingReader(f, record_size_limit=None)
            summary = reader.get_summary()
            if summary:
                for channel in summary.channels.values():
                    mcap_topics.append(channel.topic)
    except Exception:
        mcap_topics = []

    # Fallback: forward scan for truncated files without footer
    if not mcap_topics:
        from mcap.stream_reader import StreamReader
        try:
            with open(bag_path, "rb") as f:
                seen = set()
                for record in StreamReader(f, record_size_limit=None).records:
                    if hasattr(record, "topic") and record.topic not in seen:
                        seen.add(record.topic)
                        mcap_topics.append(record.topic)
        except Exception:
            pass  # read as many channels as possible before corruption

    expected_topics = set(topic_map.keys())
    mcap_topics_set = set(mcap_topics)

    return {
        "mcap_topics": mcap_topics,
        "expected_topics": list(expected_topics),
        "missing_topics": list(expected_topics - mcap_topics_set),
        "found_topics": list(expected_topics & mcap_topics_set),
    }


def extract_frames(
    bag_path: str,
    config: Rosbag,
    rot_img: bool = False,
) -> Tuple[List[Dict[str, Any]], dict[str, list[int]]]:
    """Camera-timestamp-driven sync.

    For every camera tick we keep the most recent joint_states and wrench.
    A frame is emitted only when all cameras, observation, and (when
    configured) wrench have at least one prior sample.
    """
    topic_map = config.topic_map
    joint_order = config.joint_order
    camera_names = config.camera_names
    fps = config.fps
    wrench_topic = getattr(config, "wrench_topic", "")
    shared_action_names = config.shared_action_names
    required_action_names = set(joint_order.get("action", {}).keys())
    dedicated_action_names = required_action_names.difference(shared_action_names)

    primary_cam_name = camera_names[0] if camera_names else None
    timegap = 1_000_000_000 // fps if fps > 0 else 0
    camera_sync_tolerance_ns = timegap // 2 if timegap else 0

    latest_joint_msg = None
    latest_joint_schema = ""
    latest_wrench_msg = None
    latest_wrench_schema = ""
    latest_action_msg: dict[str, Any] = {}
    latest_action_schema: dict[str, str] = {}
    latest_cam_msg = {cam: None for cam in camera_names}
    latest_cam_schema = {cam: "" for cam in camera_names}
    latest_cam_t_ns = {cam: None for cam in camera_names}

    frames: List[Dict[str, Any]] = []
    timestamps: dict[str, list[int]] = {v: [] for v in topic_map.values()}
    last_emitted_tick_ns: int | None = None
    for sa in shared_action_names:
        timestamps[sa] = []

    for topic, msg, t_ns, schema_name in _read_rosbag_messages(bag_path):
        canonical = topic_map.get(topic)
        if canonical is None:
            continue
        timestamps[canonical].append(t_ns)
        camera_updated = False

        if canonical == "observation":
            latest_joint_msg = msg
            latest_joint_schema = schema_name
            for sa in shared_action_names:
                timestamps[sa].append(t_ns)
        elif canonical == "wrench":
            latest_wrench_msg = msg
            latest_wrench_schema = schema_name
        elif canonical in camera_names:
            latest_cam_msg[canonical] = msg
            latest_cam_schema[canonical] = schema_name
            latest_cam_t_ns[canonical] = t_ns
            camera_updated = True
        elif canonical == "action" or canonical.startswith("action_"):
            latest_action_msg[canonical] = msg
            latest_action_schema[canonical] = schema_name

        if not camera_updated:
            continue

        primary_tick_ns = latest_cam_t_ns.get(primary_cam_name) if primary_cam_name else None
        if primary_tick_ns is None:
            continue
        if last_emitted_tick_ns == primary_tick_ns:
            continue
        if timegap and last_emitted_tick_ns is not None:
            if (primary_tick_ns - last_emitted_tick_ns) < timegap:
                continue
        if latest_joint_msg is None:
            continue
        if wrench_topic and latest_wrench_msg is None:
            continue
        if any(latest_cam_t_ns[cam] is None for cam in camera_names):
            continue
        if camera_sync_tolerance_ns:
            if any(
                abs(int(latest_cam_t_ns[cam]) - int(primary_tick_ns))
                > camera_sync_tolerance_ns
                for cam in camera_names
            ):
                continue
        if any(name not in latest_action_msg for name in dedicated_action_names):
            continue

        schema_map = {
            "observation": latest_joint_schema,
            **{cam: latest_cam_schema[cam] for cam in camera_names},
        }
        if wrench_topic:
            schema_map["wrench"] = latest_wrench_schema

        leader_msgs = {
            name: latest_action_msg[name] for name in dedicated_action_names
        }
        for name in dedicated_action_names:
            schema_map[name] = latest_action_schema[name]
        for sa in shared_action_names:
            leader_msgs[sa] = latest_joint_msg
            schema_map[sa] = latest_joint_schema

        frame = build_frame(
            image_msgs=dict(latest_cam_msg),
            follower_msgs={"observation": latest_joint_msg},
            leader_msgs=leader_msgs,
            joint_order=joint_order,
            rot_img=rot_img,
            schema_map=schema_map,
            wrench_msg=latest_wrench_msg if wrench_topic else None,
        )
        if frame is not None:
            frame["emitted_timestamp_ns"] = int(primary_tick_ns)
            frames.append(frame)
            last_emitted_tick_ns = primary_tick_ns

    return frames, timestamps


def build_extraction_config(
    detail: dict,
    fps: int,
    robot_type: str,
) -> Rosbag:
    """Build Rosbag extraction config from metacard fields."""
    camera_topic_map = detail["camera_topic_map"]
    joint_names = detail["joint_names"]
    action_topic_to_canonical = _resolve_action_topics(detail["action_topics_map"])
    state_topic = detail["state_topic"]
    wrench_topic = detail.get("wrench_topic", "")

    topic_map: dict[str, str] = {}
    for cam_name, topic in camera_topic_map.items():
        topic_map[topic] = cam_name
    topic_map[state_topic] = "observation"
    if wrench_topic:
        topic_map[wrench_topic] = "wrench"

    shared_action_names: list[str] = []
    for action_topic, canonical in action_topic_to_canonical.items():
        if action_topic == state_topic:
            shared_action_names.append(canonical)
        else:
            topic_map[action_topic] = canonical
    camera_names = sorted(camera_topic_map.keys())

    all_action_names = sorted(set(action_topic_to_canonical.values()))
    left = [n for n in all_action_names if "left" in n.lower() and "right" not in n.lower()]
    right = [n for n in all_action_names if "right" in n.lower() and "left" not in n.lower()]
    if left and right:
        others = [n for n in all_action_names if n not in left and n not in right]
        action_order = left + right + sorted(others)
    else:
        action_order = all_action_names

    action_joint_order = _build_action_joint_order(action_order, joint_names)

    joint_order: dict[str, Any] = {
        "obs": joint_names,
        "action": action_joint_order,
    }

    return Rosbag(
        topic_map=topic_map,
        action_order=action_order,
        joint_order=joint_order,
        camera_names=camera_names,
        fps=fps,
        hz_min_ratio=HZ_MIN_RATIO,
        robot_type=robot_type,
        shared_action_names=shared_action_names,
        wrench_topic=wrench_topic,
    )
