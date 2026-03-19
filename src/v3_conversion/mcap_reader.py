"""MCAP file I/O for the standalone conversion pipeline.

Replaces rosbag2_py with pure Python mcap + mcap-ros2-support.
All frame extraction and config assembly logic is preserved from
the original dataset_manager.conversion.mcap_reader.
"""

from typing import Any, Dict, List, Tuple

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from v3_conversion.data_spec import Rosbag
from v3_conversion.data_converter import build_frame

HZ_MIN_RATIO = 0.7


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _read_rosbag_messages(bag_path: str):
    """Yield ``(topic, msg, timestamp, schema_name)`` from an MCAP bag.

    Uses mcap-ros2-support DecoderFactory for CDR deserialization.
    schema_name comes from the MCAP channel's schema (e.g.,
    ``"sensor_msgs/msg/CompressedImage"``).
    """
    decoder = DecoderFactory()
    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[decoder])
        for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
            schema_name = schema.name if schema else ""
            yield channel.topic, decoded_msg, message.log_time, schema_name


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
    "left": ["left", "_l_", "_l"],
    "right": ["right", "_r_", "_r"],
}


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
                if any(p in name.lower() for p in patterns)
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
    """Validate that expected topics exist in MCAP file."""
    mcap_topics: list[str] = []
    with open(bag_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary:
            for channel in summary.channels.values():
                mcap_topics.append(channel.topic)

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

    frames: List[Dict[str, Any]] = []
    timestamps: dict[str, list[int]] = {name: [] for name in topic_map.values()}

    timegap = 1_000_000_000 // fps
    ts_ref = -timegap
    timing_source = camera_names[0] if camera_names else "observation"

    image_msgs: dict = {}
    follower_msgs: dict = {}
    leader_msgs: dict = {}
    schema_map: dict[str, str] = {}
    msg_flag = {name: False for name in topic_map.values()}
    cnt = 0
    timing = False

    for topic, msg, t, schema_name in _read_rosbag_messages(bag_path):
        if topic not in topic_map:
            continue

        canonical_name = topic_map[topic]
        timestamps[canonical_name].append(t)
        schema_map[canonical_name] = schema_name

        if not timing and canonical_name == timing_source:
            if (t - ts_ref) < timegap:
                continue
            ts_ref = t if ts_ref < 0 else ts_ref + timegap
            timing = True

        if not msg_flag[canonical_name]:
            if canonical_name.startswith("cam_"):
                image_msgs[canonical_name] = msg
            elif canonical_name == "action" or canonical_name.startswith("action_"):
                leader_msgs[canonical_name] = msg
            elif canonical_name == "observation":
                follower_msgs[canonical_name] = msg
            msg_flag[canonical_name] = True
            cnt += 1

        if cnt < len(msg_flag):
            continue

        frame = build_frame(
            image_msgs, follower_msgs, leader_msgs,
            joint_order, rot_img, schema_map,
        )
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
    for action_topic, canonical in action_topic_to_canonical.items():
        topic_map[action_topic] = canonical
    camera_names = sorted(camera_topic_map.keys())

    # 3. Determine action order and joint structure
    all_action_names = sorted(set(action_topic_to_canonical.values()))
    left = [n for n in all_action_names if "left" in n.lower() and "right" not in n.lower()]
    right = [n for n in all_action_names if "right" in n.lower() and "left" not in n.lower()]
    others = [n for n in all_action_names if n not in left and n not in right]
    action_order = left + right + sorted(others)

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
    )
