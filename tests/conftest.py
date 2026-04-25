"""Shared pytest fixtures for the rosbag-to-lerobot test suite."""

from pathlib import Path
import os
from typing import Any, Dict
from glob import glob

import numpy as np
import pytest
from mcap_ros2.writer import Writer as Ros2Writer

def _candidate_ros_share_dirs() -> list[Path]:
    candidates: list[Path] = []
    prefixes = []
    for env_var in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
        prefixes.extend(
            Path(entry)
            for entry in os.environ.get(env_var, "").split(os.pathsep)
            if entry
        )
    ros_distro = os.environ.get("ROS_DISTRO")
    if ros_distro:
        prefixes.append(Path("/opt/ros") / ros_distro)
    prefixes.extend(Path(path) for path in sorted(glob("/opt/ros/*")))

    seen: set[Path] = set()
    for prefix in prefixes:
        share_dir = prefix / "share"
        if share_dir.exists() and share_dir not in seen:
            seen.add(share_dir)
            candidates.append(share_dir)
    return candidates


def _load_ros2_msgdef(datatype: str) -> str:
    package, _, name = datatype.partition("/msg/")
    for share_dir in _candidate_ros_share_dirs():
        msg_path = share_dir / package / "msg" / f"{name}.msg"
        if msg_path.exists():
            return msg_path.read_text()
    raise FileNotFoundError(f"Unable to locate ROS msg definition for {datatype}")


def _build_tf_message_msgdef() -> str:
    definitions = [
        ("tf2_msgs/msg/TFMessage", _load_ros2_msgdef("tf2_msgs/msg/TFMessage")),
        (
            "geometry_msgs/msg/TransformStamped",
            _load_ros2_msgdef("geometry_msgs/msg/TransformStamped"),
        ),
        ("std_msgs/msg/Header", _load_ros2_msgdef("std_msgs/msg/Header")),
        (
            "builtin_interfaces/msg/Time",
            _load_ros2_msgdef("builtin_interfaces/msg/Time"),
        ),
        ("geometry_msgs/msg/Transform", _load_ros2_msgdef("geometry_msgs/msg/Transform")),
        ("geometry_msgs/msg/Vector3", _load_ros2_msgdef("geometry_msgs/msg/Vector3")),
        (
            "geometry_msgs/msg/Quaternion",
            _load_ros2_msgdef("geometry_msgs/msg/Quaternion"),
        ),
    ]
    return "\n================================================\n".join(
        [
            definitions[0][1],
            *[
                f"MSG: {datatype}\n{msgdef}"
                for datatype, msgdef in definitions[1:]
            ],
        ]
    )


def _build_wrench_stamped_msgdef() -> str:
    definitions = [
        (
            "geometry_msgs/msg/WrenchStamped",
            _load_ros2_msgdef("geometry_msgs/msg/WrenchStamped"),
        ),
        ("std_msgs/msg/Header", _load_ros2_msgdef("std_msgs/msg/Header")),
        (
            "builtin_interfaces/msg/Time",
            _load_ros2_msgdef("builtin_interfaces/msg/Time"),
        ),
        ("geometry_msgs/msg/Wrench", _load_ros2_msgdef("geometry_msgs/msg/Wrench")),
        ("geometry_msgs/msg/Vector3", _load_ros2_msgdef("geometry_msgs/msg/Vector3")),
    ]
    return "\n================================================\n".join(
        [
            definitions[0][1],
            *[
                f"MSG: {datatype}\n{msgdef}"
                for datatype, msgdef in definitions[1:]
            ],
        ]
    )


def _build_image_msgdef() -> str:
    definitions = [
        ("sensor_msgs/msg/Image", _load_ros2_msgdef("sensor_msgs/msg/Image")),
        ("std_msgs/msg/Header", _load_ros2_msgdef("std_msgs/msg/Header")),
        (
            "builtin_interfaces/msg/Time",
            _load_ros2_msgdef("builtin_interfaces/msg/Time"),
        ),
    ]
    return "\n================================================\n".join(
        [
            definitions[0][1],
            *[
                f"MSG: {datatype}\n{msgdef}"
                for datatype, msgdef in definitions[1:]
            ],
        ]
    )


def _build_controller_state_msgdef() -> str:
    definitions = [
        (
            "aic_control_interfaces/msg/ControllerState",
            "std_msgs/Header header\ngeometry_msgs/Pose tcp_pose",
        ),
        ("std_msgs/msg/Header", _load_ros2_msgdef("std_msgs/msg/Header")),
        (
            "builtin_interfaces/msg/Time",
            _load_ros2_msgdef("builtin_interfaces/msg/Time"),
        ),
        ("geometry_msgs/msg/Pose", _load_ros2_msgdef("geometry_msgs/msg/Pose")),
        ("geometry_msgs/msg/Point", _load_ros2_msgdef("geometry_msgs/msg/Point")),
        (
            "geometry_msgs/msg/Quaternion",
            _load_ros2_msgdef("geometry_msgs/msg/Quaternion"),
        ),
    ]
    return "\n================================================\n".join(
        [
            definitions[0][1],
            *[
                f"MSG: {datatype}\n{msgdef}"
                for datatype, msgdef in definitions[1:]
            ],
        ]
    )


def _build_motion_update_msgdef() -> str:
    definitions = [
        (
            "aic_control_interfaces/msg/MotionUpdate",
            "std_msgs/Header header\n"
            "geometry_msgs/Pose pose\n"
            "geometry_msgs/Twist velocity\n"
            "float64[] target_stiffness\n"
            "float64[] target_damping",
        ),
        ("std_msgs/msg/Header", _load_ros2_msgdef("std_msgs/msg/Header")),
        (
            "builtin_interfaces/msg/Time",
            _load_ros2_msgdef("builtin_interfaces/msg/Time"),
        ),
        ("geometry_msgs/msg/Pose", _load_ros2_msgdef("geometry_msgs/msg/Pose")),
        ("geometry_msgs/msg/Point", _load_ros2_msgdef("geometry_msgs/msg/Point")),
        (
            "geometry_msgs/msg/Quaternion",
            _load_ros2_msgdef("geometry_msgs/msg/Quaternion"),
        ),
        ("geometry_msgs/msg/Twist", _load_ros2_msgdef("geometry_msgs/msg/Twist")),
        ("geometry_msgs/msg/Vector3", _load_ros2_msgdef("geometry_msgs/msg/Vector3")),
    ]
    return "\n================================================\n".join(
        [
            definitions[0][1],
            *[
                f"MSG: {datatype}\n{msgdef}"
                for datatype, msgdef in definitions[1:]
            ],
        ]
    )


@pytest.fixture
def tmp_dataset_root(tmp_path: Path) -> Path:
    """Empty directory that acts as a LeRobot dataset root."""
    root = tmp_path / "dataset"
    root.mkdir()
    return root


@pytest.fixture
def sample_semantic_fields() -> Dict[str, Any]:
    """Semantic fields matching run_01_20260412_141241 / trial_1_score95."""
    return {
        "cable_type": "sfp_sc",
        "cable_name": "cable_0",
        "plug_type": "sfp",
        "plug_name": "sfp_tip",
        "port_type": "sfp",
        "port_name": "sfp_port_0",
        "target_module": "nic_card_mount_0",
    }


@pytest.fixture
def build_mcap_fixture(tmp_path: Path):
    """Factory that writes a tiny MCAP file with configurable topics.

    Usage:
        bag = build_mcap_fixture(
            joint_states=[(t_ns, names, positions), ...],
            joint_state_topics={"/leader_joint_states": [(t_ns, names, positions)]},
            wrench=[(t_ns, fx,fy,fz, tx,ty,tz), ...],
            images={"cam_left": [(t_ns, h, w, bytes)]},
            insertion_event=[(t_ns, "/nic_card_mount_0/sfp_port_0")],
            scoring_tf=[(t_ns, [(parent, child, x,y,z, qx,qy,qz,qw)])],
            controller_state=[(t_ns, x,y,z, qx,qy,qz,qw)],
            tf=[(t_ns, [(parent, child, x,y,z, qx,qy,qz,qw)])],
            pose_commands=[(t_ns, x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz, stiffness, damping)],
        )
    """

    def _build(
        path: Path = tmp_path / "sample.mcap",
        *,
        joint_states=None,
        joint_state_topics=None,
        wrench=None,
        images=None,
        insertion_event=None,
        scoring_tf=None,
        controller_state=None,
        tf=None,
        pose_commands=None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            writer = Ros2Writer(f)
            string_schema = None
            tf_schema = None
            pending_messages = []

            def queue_message(t_ns, priority, topic, schema, message):
                pending_messages.append((t_ns, priority, topic, schema, message))

            if joint_states or joint_state_topics:
                joint_state_msgdef = (
                    "std_msgs/Header header\nstring[] name\n"
                    "float64[] position\nfloat64[] velocity\nfloat64[] effort\n"
                    "===\n"
                    "MSG: std_msgs/Header\n"
                    "builtin_interfaces/Time stamp\n"
                    "string frame_id\n"
                    "===\n"
                    "MSG: builtin_interfaces/Time\n"
                    "uint32 sec\n"
                    "uint32 nanosec"
                )
                joint_state_schema = writer.register_msgdef(
                    datatype="sensor_msgs/msg/JointState",
                    msgdef_text=joint_state_msgdef,
                )
                joint_state_topics_to_write = {}
                if joint_states:
                    joint_state_topics_to_write["/joint_states"] = joint_states
                if joint_state_topics:
                    joint_state_topics_to_write.update(joint_state_topics)
                for topic, messages in joint_state_topics_to_write.items():
                    for item in messages:
                        if len(item) == 3:
                            t_ns, names, positions = item
                            velocities = [0.0] * len(names)
                        else:
                            t_ns, names, positions, velocities = item
                        queue_message(
                            t_ns,
                            0,
                            topic,
                            joint_state_schema,
                            {
                                "header": {
                                    "stamp": {
                                        "sec": t_ns // 1_000_000_000,
                                        "nanosec": t_ns % 1_000_000_000,
                                    },
                                    "frame_id": "base_link",
                                },
                                "name": list(names),
                                "position": list(positions),
                                "velocity": list(velocities),
                                "effort": [0.0] * len(names),
                            },
                        )
            if insertion_event:
                string_schema = string_schema or writer.register_msgdef(
                    datatype="std_msgs/msg/String",
                    msgdef_text="string data",
                )
                for t_ns, data in insertion_event:
                    queue_message(
                        t_ns, 10, "/scoring/insertion_event", string_schema, {"data": data}
                    )
            if scoring_tf:
                tf_schema = tf_schema or writer.register_msgdef(
                    datatype="tf2_msgs/msg/TFMessage",
                    msgdef_text=_build_tf_message_msgdef(),
                )
                for t_ns, transforms in scoring_tf:
                    tf_msgs = []
                    for parent, child, x, y, z, qx, qy, qz, qw in transforms:
                        tf_msgs.append(
                            {
                                "header": {
                                    "stamp": {
                                        "sec": t_ns // 1_000_000_000,
                                        "nanosec": t_ns % 1_000_000_000,
                                    },
                                    "frame_id": parent,
                                },
                                "child_frame_id": child,
                                "transform": {
                                    "translation": {"x": x, "y": y, "z": z},
                                    "rotation": {
                                        "x": qx,
                                        "y": qy,
                                        "z": qz,
                                        "w": qw,
                                    },
                                },
                            }
                        )
                    queue_message(
                        t_ns,
                        20,
                        "/scoring/tf",
                        tf_schema,
                        {"transforms": tf_msgs},
                    )
            if controller_state:
                controller_schema = writer.register_msgdef(
                    datatype="aic_control_interfaces/msg/ControllerState",
                    msgdef_text=_build_controller_state_msgdef(),
                )
                for t_ns, x, y, z, qx, qy, qz, qw in controller_state:
                    queue_message(
                        t_ns,
                        30,
                        "/aic_controller/controller_state",
                        controller_schema,
                        {
                            "header": {
                                "stamp": {
                                    "sec": t_ns // 1_000_000_000,
                                    "nanosec": t_ns % 1_000_000_000,
                                },
                                "frame_id": "base_link",
                            },
                            "tcp_pose": {
                                "position": {"x": x, "y": y, "z": z},
                                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                            },
                        },
                    )
            if tf:
                tf_schema = tf_schema or writer.register_msgdef(
                    datatype="tf2_msgs/msg/TFMessage",
                    msgdef_text=_build_tf_message_msgdef(),
                )
                for t_ns, transforms in tf:
                    tf_msgs = []
                    for parent, child, x, y, z, qx, qy, qz, qw in transforms:
                        tf_msgs.append(
                            {
                                "header": {
                                    "stamp": {
                                        "sec": t_ns // 1_000_000_000,
                                        "nanosec": t_ns % 1_000_000_000,
                                    },
                                    "frame_id": parent,
                                },
                                "child_frame_id": child,
                                "transform": {
                                    "translation": {"x": x, "y": y, "z": z},
                                    "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
                                },
                            }
                        )
                    queue_message(t_ns, 40, "/tf", tf_schema, {"transforms": tf_msgs})
            if pose_commands:
                motion_schema = writer.register_msgdef(
                    datatype="aic_control_interfaces/msg/MotionUpdate",
                    msgdef_text=_build_motion_update_msgdef(),
                )
                for (
                    t_ns,
                    x, y, z, qx, qy, qz, qw,
                    vx, vy, vz, wx, wy, wz,
                    stiffness,
                    damping,
                ) in pose_commands:
                    queue_message(
                        t_ns,
                        50,
                        "/aic_controller/pose_commands",
                        motion_schema,
                        {
                            "header": {
                                "stamp": {
                                    "sec": t_ns // 1_000_000_000,
                                    "nanosec": t_ns % 1_000_000_000,
                                },
                                "frame_id": "base_link",
                            },
                            "pose": {
                                "position": {"x": x, "y": y, "z": z},
                                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                            },
                            "velocity": {
                                "linear": {"x": vx, "y": vy, "z": vz},
                                "angular": {"x": wx, "y": wy, "z": wz},
                            },
                            "target_stiffness": list(stiffness),
                            "target_damping": list(damping),
                        },
                    )
            if wrench:
                wrench_schema = writer.register_msgdef(
                    datatype="geometry_msgs/msg/WrenchStamped",
                    msgdef_text=_build_wrench_stamped_msgdef(),
                )
                for t_ns, fx, fy, fz, tx, ty, tz in wrench:
                    queue_message(
                        t_ns,
                        5,
                        "/fts_broadcaster/wrench",
                        wrench_schema,
                        {
                            "header": {
                                "stamp": {
                                    "sec": t_ns // 1_000_000_000,
                                    "nanosec": t_ns % 1_000_000_000,
                                },
                                "frame_id": "tool_link",
                            },
                            "wrench": {
                                "force": {"x": fx, "y": fy, "z": fz},
                                "torque": {"x": tx, "y": ty, "z": tz},
                            },
                        },
                    )
            if images:
                image_schema = writer.register_msgdef(
                    datatype="sensor_msgs/msg/Image",
                    msgdef_text=_build_image_msgdef(),
                )
                ordered_image_frames = []
                for topic_rank, topic in enumerate(reversed(list(images.keys()))):
                    for t_ns, height, width, data_bytes in images[topic]:
                        ordered_image_frames.append(
                            (t_ns, topic_rank, topic, height, width, data_bytes)
                        )
                ordered_image_frames.sort(key=lambda item: (item[0], item[1]))
                for t_ns, topic_rank, topic, height, width, data_bytes in ordered_image_frames:
                    queue_message(
                        t_ns,
                        100 + topic_rank,
                        topic,
                        image_schema,
                        {
                            "header": {
                                "stamp": {
                                    "sec": t_ns // 1_000_000_000,
                                    "nanosec": t_ns % 1_000_000_000,
                                },
                                "frame_id": "camera",
                            },
                            "height": height,
                            "width": width,
                            "encoding": "rgb8",
                            "is_bigendian": 0,
                            "step": width * 3,
                            "data": list(data_bytes),
                        },
                    )
            for t_ns, _, topic, schema, message in sorted(
                pending_messages, key=lambda item: (item[0], item[1])
            ):
                writer.write_message(
                    topic=topic,
                    schema=schema,
                    message=message,
                    log_time=t_ns,
                    publish_time=t_ns,
                )
            writer.finish()
        return path

    return _build
