"""Pure data-conversion helpers (no I/O, no ROS2).

Every function in this module converts **in-memory** deserialized MCAP
message objects or numpy arrays — it never touches the filesystem.

Message type dispatch uses schema-name strings from MCAP channel metadata
instead of ROS2 isinstance() checks.
"""

from typing import Any, Dict, List

import cv2
import numpy as np


# ------------------------------------------------------------------
# Schema-name dispatch handlers for joint/motion messages
# ------------------------------------------------------------------

def _handle_joint_trajectory(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle trajectory_msgs/msg/JointTrajectory."""
    joint_pos_map = dict(zip(msg_data.joint_names, msg_data.points[0].positions))
    ordered = [joint_pos_map[name] for name in joint_order]
    return np.array(ordered, dtype=np.float32)


def _handle_joint_state(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle sensor_msgs/msg/JointState."""
    joint_pos_map = dict(zip(msg_data.name, msg_data.position))
    missing = [name for name in joint_order if name not in joint_pos_map]
    if missing:
        raise KeyError(f"Missing joints in JointState: {missing}")
    ordered = [joint_pos_map[name] for name in joint_order]
    return np.array(ordered, dtype=np.float32)


def _handle_odometry(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle nav_msgs/msg/Odometry."""
    linear = np.array(
        [msg_data.twist.twist.linear.x, msg_data.twist.twist.linear.y],
        dtype=np.float32,
    )
    angular = np.array([msg_data.twist.twist.angular.z], dtype=np.float32)
    return np.concatenate((linear, angular))


def _handle_twist(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle geometry_msgs/msg/Twist."""
    linear = np.array(
        [msg_data.linear.x, msg_data.linear.y], dtype=np.float32
    )
    angular = np.array([msg_data.angular.z], dtype=np.float32)
    return np.concatenate((linear, angular))


_JOINT_HANDLERS = {
    "trajectory_msgs/msg/JointTrajectory": _handle_joint_trajectory,
    "sensor_msgs/msg/JointState": _handle_joint_state,
    "nav_msgs/msg/Odometry": _handle_odometry,
    "geometry_msgs/msg/Twist": _handle_twist,
}


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _compressed_image2cvmat(
    msg: Any, desired_encoding: str = "passthrough"
) -> np.ndarray:
    """Deserialized CompressedImage -> numpy BGR/RGB array."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if desired_encoding in ("passthrough", "raw"):
        flag = cv2.IMREAD_UNCHANGED
    else:
        flag = cv2.IMREAD_COLOR

    img = cv2.imdecode(buf, flag)
    if img is None:
        raise RuntimeError(
            f"cv2.imdecode failed (format={getattr(msg, 'format', None)})"
        )

    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if desired_encoding.lower() in ("rgb8", "rgb"):
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def _convert_joint_msg(
    msg_data: Any,
    joint_order: List[str] | None,
    schema_name: str,
) -> np.ndarray:
    """Route a deserialized joint/motion message to a numpy float32 array.

    Uses schema-name dispatch instead of isinstance() checks.
    """
    handler = _JOINT_HANDLERS.get(schema_name)
    if handler is None:
        raise ValueError(f"Unsupported message schema: {schema_name}")
    return handler(msg_data, joint_order)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def build_frame(
    image_msgs: dict,
    follower_msgs: dict,
    leader_msgs: dict,
    joint_order: Dict[str, Any],
    rot_img: bool,
    schema_map: Dict[str, str],
) -> Dict[str, Any] | None:
    """Convert accumulated deserialized messages into a single frame dict.

    Returns ``None`` when action data is incomplete.

    Parameters
    ----------
    schema_map : dict[str, str]
        Maps canonical name -> MCAP schema name (e.g.,
        ``"observation" -> "sensor_msgs/msg/JointState"``).
    """
    # -- images --
    camera_data = {}
    for key, value in (image_msgs or {}).items():
        camera_data[key] = cv2.cvtColor(
            _compressed_image2cvmat(value), cv2.COLOR_BGR2RGB
        )

    # -- observation (follower) --
    follower_data: list = []
    for canon_name, value in (follower_msgs or {}).items():
        if value is not None:
            obs_schema = schema_map.get(canon_name, "")
            follower_data.extend(
                _convert_joint_msg(value, joint_order["obs"], obs_schema)
            )

    # -- action (leader) --
    leader_joint_order = joint_order.get("action")
    if leader_msgs is None or leader_joint_order is None:
        return None

    action_data: dict = {}
    for key, order in leader_joint_order.items():
        if key not in leader_msgs or leader_msgs[key] is None:
            return None
        action_schema = schema_map.get(key, "")
        action_data[key] = _convert_joint_msg(leader_msgs[key], order, action_schema)

    # -- rotate wrist images --
    if rot_img:
        for k, v in camera_data.items():
            if "wrist" in k:
                camera_data[k] = v[::-1, ::-1].copy()

    return {"images": camera_data, "obs": follower_data, "action": action_data}


def frames_to_episode(
    frames: List[Dict[str, Any]],
    action_order: List[str],
    camera_names: List[str],
    task: str = "default_task",
):
    """Convert frame list to numpy episode dict."""
    obs_list = []
    action_lists = {key: [] for key in action_order}
    camera_lists = {cam: [] for cam in camera_names}

    for f in frames:
        obs_list.append(np.asarray(f["obs"], dtype=np.float32))

        action = f["action"]
        for key in action_order:
            action_lists[key].append(np.asarray(action[key], dtype=np.float32))

        imgs = f["images"]
        for cam in camera_names:
            if cam in imgs:
                camera_lists[cam].append(imgs[cam])

    episode = {
        "obs": np.stack(obs_list, axis=0),
        "images": camera_lists,
        "task": task,
    }
    for key in action_order:
        episode[key] = np.stack(action_lists[key], axis=0)

    return episode
