"""Pure data-conversion helpers (no I/O, no ROS2).

Every function in this module converts **in-memory** deserialized MCAP
message objects or numpy arrays — it never touches the filesystem.

Message type dispatch uses schema-name strings from MCAP channel metadata
instead of ROS2 isinstance() checks.
"""

from typing import Any, Dict, List, Optional

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
    """Handle sensor_msgs/msg/JointState — positions only."""
    joint_pos_map = dict(zip(msg_data.name, msg_data.position))
    missing = [name for name in joint_order if name not in joint_pos_map]
    if missing:
        raise KeyError(f"Missing joints in JointState: {missing}")
    ordered = [joint_pos_map[name] for name in joint_order]
    return np.array(ordered, dtype=np.float32)


def _handle_joint_state_full(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle sensor_msgs/msg/JointState — positions + velocities + efforts."""
    joint_pos_map = dict(zip(msg_data.name, msg_data.position))
    missing = [name for name in joint_order if name not in joint_pos_map]
    if missing:
        raise KeyError(f"Missing joints in JointState: {missing}")
    result = [joint_pos_map[name] for name in joint_order]

    if msg_data.velocity and len(msg_data.velocity) >= len(msg_data.name):
        vel_map = dict(zip(msg_data.name, msg_data.velocity))
        result += [vel_map[name] for name in joint_order]

    if msg_data.effort and len(msg_data.effort) >= len(msg_data.name):
        eff_map = dict(zip(msg_data.name, msg_data.effort))
        result += [eff_map[name] for name in joint_order]

    return np.array(result, dtype=np.float32)


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


def _handle_joint_motion_update(msg_data, joint_order: List[str]) -> np.ndarray:
    """Handle aic_control_interfaces/msg/JointMotionUpdate.

    Extracts joint positions from the ``target_state`` field
    (trajectory_msgs/JointTrajectoryPoint).
    """
    positions = list(msg_data.target_state.positions)
    if joint_order and len(positions) != len(joint_order):
        raise ValueError(
            f"JointMotionUpdate.target_state has {len(positions)} "
            f"positions but joint_order expects {len(joint_order)}"
        )
    return np.array(positions, dtype=np.float32)


_JOINT_HANDLERS = {
    "trajectory_msgs/msg/JointTrajectory": _handle_joint_trajectory,
    "sensor_msgs/msg/JointState": _handle_joint_state,
    "nav_msgs/msg/Odometry": _handle_odometry,
    "geometry_msgs/msg/Twist": _handle_twist,
    "aic_control_interfaces/msg/ControllerState": _handle_controller_state,
    "aic_control_interfaces/msg/JointMotionUpdate": _handle_joint_motion_update,
}


# ------------------------------------------------------------------
# Extra message handlers for v2 dataset schema
# ------------------------------------------------------------------

def _handle_wrench_stamped(msg, joint_order: List[str] = None) -> np.ndarray:
    """Handle geometry_msgs/msg/WrenchStamped.

    Extracts force (x, y, z) and torque (x, y, z).
    ``joint_order`` is accepted for interface consistency but not used.
    """
    w = msg.wrench
    return np.array(
        [w.force.x, w.force.y, w.force.z,
         w.torque.x, w.torque.y, w.torque.z],
        dtype=np.float32,
    )


def _handle_controller_state_full(msg) -> Dict[str, np.ndarray]:
    """Handle aic_control_interfaces/msg/ControllerState (full extraction).

    Returns a dict with tcp_pose, tcp_velocity, tcp_error,
    reference_tcp_pose, and target_mode arrays.
    """
    def _pose_to_array(pose):
        p = pose.position
        o = pose.orientation
        return np.array(
            [p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32
        )

    def _twist_to_array(twist):
        l = twist.linear
        a = twist.angular
        return np.array(
            [l.x, l.y, l.z, a.x, a.y, a.z], dtype=np.float32
        )

    return {
        "tcp_pose": _pose_to_array(msg.tcp_pose),
        "tcp_velocity": _twist_to_array(msg.tcp_velocity),
        "tcp_error": np.array(msg.tcp_error, dtype=np.float32),
        "reference_tcp_pose": _pose_to_array(msg.reference_tcp_pose),
        "target_mode": np.array([msg.target_mode.mode], dtype=np.float32),
    }



def _extract_plug_pose(tf_msg, plug_frame: str) -> Optional[np.ndarray]:
    """Extract plug pose from TFMessage by matching child_frame_id.

    Returns position(3) + quaternion(4) as float32[7], or None if not found.
    """
    if not plug_frame:
        return None
    for t in getattr(tf_msg, 'transforms', []):
        if plug_frame in t.child_frame_id:
            tr = t.transform.translation
            ro = t.transform.rotation
            return np.array(
                [tr.x, tr.y, tr.z, ro.x, ro.y, ro.z, ro.w], dtype=np.float32)
    return None


def _handle_string_msg(msg) -> str:
    """Handle std_msgs/msg/String."""
    return msg.data


def _handle_contacts_msg(msg) -> bool:
    """Handle contact messages.

    Returns True if any contacts are present, False otherwise.
    """
    contacts = getattr(msg, "contacts", getattr(msg, "states", []))
    return len(contacts) > 0


_EXTRA_HANDLERS = {
    "geometry_msgs/msg/WrenchStamped": _handle_wrench_stamped,
    "aic_control_interfaces/msg/ControllerState": _handle_controller_state_full,
    # TF handled directly in build_frame (needs target_frame_id arg)
    "std_msgs/msg/String": _handle_string_msg,
    "ros_gz_interfaces/msg/Contacts": _handle_contacts_msg,
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
    extra_obs_msgs: Optional[Dict[str, Any]] = None,
    event_msgs: Optional[Dict[str, Any]] = None,
    extra_schema_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any] | None:
    """Convert accumulated deserialized messages into a single frame dict.

    Returns ``None`` when action data is incomplete.

    Parameters
    ----------
    schema_map : dict[str, str]
        Maps canonical name -> MCAP schema name (e.g.,
        ``"observation" -> "sensor_msgs/msg/JointState"``).
    extra_obs_msgs : dict, optional
        Additional observation messages keyed by canonical name, processed
        via ``_EXTRA_HANDLERS``.
    event_msgs : dict, optional
        Event messages keyed by canonical name, processed via
        ``_EXTRA_HANDLERS``.
    extra_schema_map : dict, optional
        Maps canonical name -> MCAP schema name for extra/event messages.
    """
    # -- images (schema-based dispatch: raw Image or CompressedImage) --
    camera_data = {}
    for key, value in (image_msgs or {}).items():
        img_schema = schema_map.get(key, "")
        camera_data[key] = cv2.cvtColor(
            _decode_image(value, img_schema), cv2.COLOR_BGR2RGB
        )

    # -- observation (follower) — use full state (pos+vel+effort) when available --
    follower_arrays: list = []
    for canon_name, value in (follower_msgs or {}).items():
        if value is not None:
            obs_schema = schema_map.get(canon_name, "")
            if obs_schema == "sensor_msgs/msg/JointState":
                follower_arrays.append(
                    _handle_joint_state_full(value, joint_order["obs"])
                )
            else:
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

    # -- extra observations (v2 schema) --
    if extra_obs_msgs:
        _extra_map = extra_schema_map or {}
        extra_obs_data: Dict[str, Any] = {}
        for canon_name, value in extra_obs_msgs.items():
            if value is None:
                continue
            e_schema = _extra_map.get(canon_name, "")
            # TF messages need special handling (target_frame_id required)
            if e_schema == "tf2_msgs/msg/TFMessage":
                plug_frame = (extra_schema_map or {}).get("_plug_frame", "")
                pose = _extract_plug_pose(value, plug_frame)
                if pose is not None:
                    extra_obs_data["plug_pose"] = pose
                continue
            handler = _EXTRA_HANDLERS.get(e_schema)
            if handler is None:
                continue  # skip unknown schemas gracefully
            result_val = handler(value)
            # If handler returns a dict (e.g. controller_state_full), flatten into extra_obs
            if isinstance(result_val, dict):
                for sub_key, sub_val in result_val.items():
                    extra_obs_data[sub_key] = sub_val
            else:
                extra_obs_data[canon_name] = result_val
        if extra_obs_data:
            result["extra_obs"] = extra_obs_data

    # -- events (v2 schema) --
    if event_msgs:
        _extra_map = extra_schema_map or {}
        events_data: Dict[str, Any] = {}
        for canon_name, value in event_msgs.items():
            if value is None:
                continue
            e_schema = _extra_map.get(canon_name, "")
            handler = _EXTRA_HANDLERS.get(e_schema)
            if handler is None:
                continue  # skip unknown schemas gracefully
            events_data[canon_name] = handler(value)
        if events_data:
            result["events"] = events_data

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

    # Discover extra_obs and events keys from the first frame that has them
    extra_obs_keys: List[str] = []
    events_keys: List[str] = []
    for f in frames:
        if "extra_obs" in f and not extra_obs_keys:
            extra_obs_keys = list(f["extra_obs"].keys())
        if "events" in f and not events_keys:
            events_keys = list(f["events"].keys())
        if extra_obs_keys and events_keys:
            break

    extra_obs_lists: Dict[str, list] = {k: [] for k in extra_obs_keys}
    events_lists: Dict[str, list] = {k: [] for k in events_keys}

    for f in frames:
        obs_list.append(np.asarray(f["obs"], dtype=np.float32))

        action = f["action"]
        for key in action_order:
            action_lists[key].append(np.asarray(action[key], dtype=np.float32))

        imgs = f["images"]
        for cam in camera_names:
            if cam in imgs:
                camera_lists[cam].append(imgs[cam])

        # Collect extra_obs
        if extra_obs_keys:
            extra_obs = f.get("extra_obs", {})
            for k in extra_obs_keys:
                if k in extra_obs:
                    extra_obs_lists[k].append(extra_obs[k])

        # Collect events
        if events_keys:
            events = f.get("events", {})
            for k in events_keys:
                if k in events:
                    events_lists[k].append(events[k])

    episode = {
        "obs": np.stack(obs_list, axis=0),
        "images": camera_lists,
        "task": task,
    }

    for key in action_order:
        episode[key] = np.stack(action_lists[key], axis=0).astype(np.float32)

    # Stack extra_obs arrays into episode
    if extra_obs_keys:
        extra_obs_episode: Dict[str, Any] = {}
        for k, vals in extra_obs_lists.items():
            if not vals:
                continue
            if isinstance(vals[0], dict):
                # Dict-valued handler (e.g. _handle_controller_state_full)
                sub_keys = vals[0].keys()
                extra_obs_episode[k] = {
                    sk: np.stack([v[sk] for v in vals], axis=0)
                    for sk in sub_keys
                }
            elif isinstance(vals[0], np.ndarray):
                extra_obs_episode[k] = np.stack(vals, axis=0)
            else:
                extra_obs_episode[k] = vals
        if extra_obs_episode:
            episode["extra_obs"] = extra_obs_episode

    # Collect events into episode
    if events_keys:
        events_episode: Dict[str, Any] = {}
        for k, vals in events_lists.items():
            if not vals:
                continue
            if isinstance(vals[0], np.ndarray):
                events_episode[k] = np.stack(vals, axis=0)
            else:
                events_episode[k] = vals
        if events_episode:
            episode["events"] = events_episode

    return episode
