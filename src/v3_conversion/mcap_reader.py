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
    """Extract synchronised frames from MCAP.

    Returns ``(frames, timestamps)`` — Hz validation is the caller's
    responsibility.
    """
    topic_map = config.topic_map
    joint_order = config.joint_order
    camera_names = config.camera_names
    fps = config.fps

    shared_action_names = config.shared_action_names

    frames: List[Dict[str, Any]] = []
    timestamps: dict[str, list[int]] = {name: [] for name in topic_map.values()}
    # Add timestamp tracking for shared actions (not in topic_map)
    for sa in shared_action_names:
        timestamps[sa] = []

    timegap = 1_000_000_000 // fps
    ts_ref = -timegap
    timing_source = camera_names[0] if camera_names else "observation"

    image_msgs: dict = {}
    follower_msgs: dict = {}
    leader_msgs: dict = {}
    schema_map: dict[str, str] = {}
    msg_flag = {name: False for name in topic_map.values()}
    # Add flags for shared actions
    for sa in shared_action_names:
        msg_flag[sa] = False
    cnt = 0
    timing = False

    for topic, msg, t, schema_name in _read_rosbag_messages(bag_path):
        if topic not in topic_map:
            continue

        canonical_name = topic_map[topic]
        timestamps[canonical_name].append(t)
        schema_map[canonical_name] = schema_name

        # Timing gate: only the timing source controls frame pacing.
        # When the timing source arrives too early, skip *only that message*
        # (not the entire loop iteration) so other topics are still collected.
        if not timing and canonical_name == timing_source:
            if (t - ts_ref) < timegap:
                continue
            ts_ref = t if ts_ref < 0 else ts_ref + timegap
            timing = True
        elif not timing and canonical_name != timing_source:
            # Non-timing-source messages: collect them but don't advance frame
            if not msg_flag[canonical_name]:
                if canonical_name.startswith("cam_"):
                    image_msgs[canonical_name] = msg
                elif canonical_name == "action" or canonical_name.startswith("action_") or canonical_name == "wrench":
                    leader_msgs[canonical_name] = msg
                elif canonical_name == "observation":
                    follower_msgs[canonical_name] = msg
                    for sa in shared_action_names:
                        if not msg_flag[sa]:
                            leader_msgs[sa] = msg
                            timestamps[sa].append(t)
                            schema_map[sa] = schema_name
                            msg_flag[sa] = True
                            cnt += 1
                msg_flag[canonical_name] = True
                cnt += 1
            continue

        # Convention: camera keys must start with "cam_" in camera_topic_map.
        # Other canonical names: "observation" (state), "action"/"action_*" (leader).
        if not msg_flag[canonical_name]:
            if canonical_name.startswith("cam_"):
                image_msgs[canonical_name] = msg
            elif canonical_name == "action" or canonical_name.startswith("action_") or canonical_name == "wrench":
                leader_msgs[canonical_name] = msg
            elif canonical_name == "observation":
                follower_msgs[canonical_name] = msg
                # Auto-fill shared actions that use the same topic as state
                for sa in shared_action_names:
                    if not msg_flag[sa]:
                        leader_msgs[sa] = msg
                        timestamps[sa].append(t)
                        schema_map[sa] = schema_name
                        msg_flag[sa] = True
                        cnt += 1
            msg_flag[canonical_name] = True
            cnt += 1

        if cnt < len(msg_flag):
            continue

        frame = build_frame(
            image_msgs, follower_msgs, leader_msgs,
            joint_order, rot_img, schema_map,
        )
        if frame is not None:
            frames.append(frame)

        for key in msg_flag:
            msg_flag[key] = False
        cnt = 0
        timing = False

    return frames, timestamps


def build_extraction_config(
    detail: dict,
    fps: int,
    robot_type: str,
) -> Rosbag:
    """Build Rosbag extraction config from metacard fields."""
    camera_topic_map = detail["camera_topic_map"]
    joint_names = detail["joint_names"]

    # 1. Resolve action topics and canonical names
    action_topic_to_canonical = _resolve_action_topics(
        detail["action_topics_map"]
    )

    # 2. Build unified topic map
    state_topic = detail["state_topic"]
    topic_map: dict[str, str] = {}
    for cam_name, topic in camera_topic_map.items():
        topic_map[topic] = cam_name
    topic_map[state_topic] = "observation"

    # Track action canonical names that share the same topic as state
    shared_action_names: list[str] = []
    for action_topic, canonical in action_topic_to_canonical.items():
        if action_topic == state_topic:
            shared_action_names.append(canonical)
        else:
            topic_map[action_topic] = canonical
    camera_names = sorted(camera_topic_map.keys())

    # 3. Determine action order and joint structure
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
    )
