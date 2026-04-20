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
    if not msg_data.points:
        raise ValueError("JointTrajectory has empty points list")
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
    """Handle nav_msgs/msg/Odometry.

    Extracts [linear.x, linear.y, angular.z].  ``joint_order`` is accepted
    for interface consistency but not used for field selection.
    """
    linear = np.array(
        [msg_data.twist.twist.linear.x, msg_data.twist.twist.linear.y],
        dtype=np.float32,
    )
    angular = np.array([msg_data.twist.twist.angular.z], dtype=np.float32)
    result = np.concatenate((linear, angular))
    if joint_order and len(joint_order) != len(result):
        raise ValueError(
            f"Odometry produces {len(result)} values but joint_order "
            f"expects {len(joint_order)}: {joint_order}"
        )
    return result


def _handle_twist(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle geometry_msgs/msg/Twist.

    Extracts [linear.x, linear.y, angular.z].  ``joint_order`` is accepted
    for interface consistency but not used for field selection.
    """
    linear = np.array(
        [msg_data.linear.x, msg_data.linear.y], dtype=np.float32
    )
    angular = np.array([msg_data.angular.z], dtype=np.float32)
    result = np.concatenate((linear, angular))
    if joint_order and len(joint_order) != len(result):
        raise ValueError(
            f"Twist produces {len(result)} values but joint_order "
            f"expects {len(joint_order)}: {joint_order}"
        )
    return result


def _handle_controller_state(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle aic_control_interfaces/msg/ControllerState.

    Extracts joint positions from the ``reference_joint_state`` field
    (trajectory_msgs/JointTrajectoryPoint).  Since JointTrajectoryPoint
    carries no joint names, positions are returned in their original order
    and ``joint_order`` is used only for length validation.
    """
    positions = list(msg_data.reference_joint_state.positions)
    if joint_order and len(positions) != len(joint_order):
        raise ValueError(
            f"ControllerState.reference_joint_state has {len(positions)} "
            f"positions but joint_order expects {len(joint_order)}"
    )
    return np.array(positions, dtype=np.float32)


def _handle_wrench_stamped(msg_data, joint_order) -> np.ndarray:
    """Handle geometry_msgs/msg/WrenchStamped -> [Fx,Fy,Fz,Tx,Ty,Tz]."""
    w = msg_data.wrench
    result = np.array(
        [
            w.force.x,
            w.force.y,
            w.force.z,
            w.torque.x,
            w.torque.y,
            w.torque.z,
        ],
        dtype=np.float32,
    )
    if joint_order and len(joint_order) != len(result):
        raise ValueError(
            f"WrenchStamped produces {len(result)} values but joint_order "
            f"expects {len(joint_order)}: {joint_order}"
        )
    return result


_JOINT_HANDLERS = {
    "trajectory_msgs/msg/JointTrajectory": _handle_joint_trajectory,
    "sensor_msgs/msg/JointState": _handle_joint_state,
    "nav_msgs/msg/Odometry": _handle_odometry,
    "geometry_msgs/msg/Twist": _handle_twist,
    "aic_control_interfaces/msg/ControllerState": _handle_controller_state,
    "geometry_msgs/msg/WrenchStamped": _handle_wrench_stamped,
}


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _compressed_image2cvmat(msg: Any) -> np.ndarray:
    """Deserialized CompressedImage -> numpy BGR array."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(
            f"cv2.imdecode failed (format={getattr(msg, 'format', None)})"
        )

    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Ensure 3-channel BGR output
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# Encoding -> number of channels for raw Image messages
_RAW_ENCODING_CHANNELS = {
    "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4,
    "mono8": 1, "mono16": 1,
    "8UC1": 1, "8UC3": 3, "8UC4": 4,
    "16UC1": 1, "16UC3": 3, "32FC1": 1,
}


def _raw_image2cvmat(msg: Any) -> np.ndarray:
    """Deserialized sensor_msgs/Image -> numpy BGR array."""
    encoding = getattr(msg, "encoding", "bgr8")
    height = msg.height
    width = msg.width
    channels = _RAW_ENCODING_CHANNELS.get(encoding)
    if channels is None:
        raise ValueError(
            f"Unsupported raw image encoding: '{encoding}'. "
            f"Supported: {list(_RAW_ENCODING_CHANNELS.keys())}"
        )

    dtype = np.uint8
    if "32F" in encoding:
        dtype = np.float32
    elif "16" in encoding:
        dtype = np.uint16

    buf = np.frombuffer(bytes(msg.data), dtype=dtype)
    if channels == 1:
        img = buf.reshape(height, width)
    else:
        img = buf.reshape(height, width, channels)

    # Normalize to 8-bit BGR
    if dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if encoding in ("rgb8",):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding in ("rgba8",):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif encoding in ("bgra8",):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def _decode_image(msg: Any, schema_name: str) -> np.ndarray:
    """Dispatch image decoding based on schema name. Returns BGR array."""
    if "CompressedImage" in schema_name:
        return _compressed_image2cvmat(msg)
    else:
        return _raw_image2cvmat(msg)


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
    wrench_msg: Any | None = None,
) -> Dict[str, Any] | None:
    """Convert accumulated deserialized messages into a single frame dict.

    Returns ``None`` when action data is incomplete.

    Parameters
    ----------
    schema_map : dict[str, str]
        Maps canonical name -> MCAP schema name (e.g.,
        ``"observation" -> "sensor_msgs/msg/JointState"``).
    """
    # -- images (schema-based dispatch: raw Image or CompressedImage) --
    camera_data = {}
    for key, value in (image_msgs or {}).items():
        img_schema = schema_map.get(key, "")
        camera_data[key] = cv2.cvtColor(
            _decode_image(value, img_schema), cv2.COLOR_BGR2RGB
        )

    # -- observation (follower) --
    follower_arrays: list = []
    for canon_name, value in (follower_msgs or {}).items():
        if value is not None:
            obs_schema = schema_map.get(canon_name, "")
            follower_arrays.append(
                _convert_joint_msg(value, joint_order["obs"], obs_schema)
            )
    follower_data = np.concatenate(follower_arrays) if follower_arrays else np.array([], dtype=np.float32)

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

    result = {"images": camera_data, "obs": follower_data, "action": action_data}
    if wrench_msg is not None:
        wrench_schema = schema_map.get("wrench", "geometry_msgs/msg/WrenchStamped")
        result["wrench"] = _convert_joint_msg(wrench_msg, None, wrench_schema)
    return result


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
    wrench_list: list = []

    # Pop frames from the input list to free memory as we consume them
    while frames:
        f = frames.pop(0)
        obs_list.append(np.asarray(f["obs"], dtype=np.float32))

        action = f["action"]
        for key in action_order:
            action_lists[key].append(np.asarray(action[key], dtype=np.float32))

        imgs = f["images"]
        for cam in camera_names:
            if cam in imgs:
                camera_lists[cam].append(imgs[cam])

        if "wrench" in f:
            wrench_list.append(np.asarray(f["wrench"], dtype=np.float32))

        f.clear()

    if wrench_list and len(wrench_list) != len(obs_list):
        raise ValueError(
            "frames_to_episode requires wrench to be present in every frame "
            "or in none of them"
        )

    episode = {
        "obs": np.stack(obs_list, axis=0),
        "images": camera_lists,
        "task": task,
    }
    del obs_list
    for key in action_order:
        episode[key] = np.stack(action_lists[key], axis=0)
    del action_lists

    if wrench_list:
        episode["wrench"] = np.stack(wrench_list, axis=0)

    return episode
