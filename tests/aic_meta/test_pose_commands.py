import os
from glob import glob
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from mcap.writer import Writer as McapWriter
from mcap_ros2.writer import SchemaEncoding, serialize_dynamic

from v3_conversion.aic_meta.pose_commands import extract_pose_commands
from v3_conversion.aic_meta.writer import write_pose_commands_parquet


def _load_ros2_msgdef(datatype: str) -> str:
    package, _, name = datatype.partition("/msg/")
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

    for prefix in prefixes:
        msg_path = prefix / "share" / package / "msg" / f"{name}.msg"
        if msg_path.exists():
            return msg_path.read_text()
    raise FileNotFoundError(f"Unable to locate ROS msg definition for {datatype}")


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


def _build_mixed_pose_command_mcap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer = McapWriter(f)
        writer.start(profile="ros2", library="test")
        string_schema_id = writer.register_schema(
            name="std_msgs/msg/String",
            encoding=SchemaEncoding.ROS2,
            data=b"string data",
        )
        motion_msgdef = _build_motion_update_msgdef()
        motion_schema_id = writer.register_schema(
            name="aic_control_interfaces/msg/MotionUpdate",
            encoding=SchemaEncoding.ROS2,
            data=motion_msgdef.encode(),
        )
        string_channel_id = writer.register_channel(
            topic="/aic_controller/pose_commands",
            message_encoding="cdr",
            schema_id=string_schema_id,
        )
        motion_channel_id = writer.register_channel(
            topic="/aic_controller/pose_commands",
            message_encoding="cdr",
            schema_id=motion_schema_id,
        )
        string_encoder = serialize_dynamic("std_msgs/msg/String", "string data")[
            "std_msgs/msg/String"
        ]
        motion_encoder = serialize_dynamic(
            "aic_control_interfaces/msg/MotionUpdate",
            motion_msgdef,
        )["aic_control_interfaces/msg/MotionUpdate"]
        writer.add_message(
            channel_id=string_channel_id,
            data=string_encoder({"data": "not a motion update"}),
            log_time=1_000_000_000,
            publish_time=1_000_000_000,
        )
        writer.add_message(
            channel_id=motion_channel_id,
            data=motion_encoder(
                {
                    "header": {
                        "stamp": {"sec": 1, "nanosec": 500_000_000},
                        "frame_id": "base_link",
                    },
                    "pose": {
                        "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0,
                        },
                    },
                    "velocity": {
                        "linear": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "angular": {"x": 0.4, "y": 0.5, "z": 0.6},
                    },
                    "target_stiffness": [91.0] * 36,
                    "target_damping": [21.0] * 36,
                }
            ),
            log_time=1_500_000_000,
            publish_time=1_500_000_000,
        )
        writer.finish()
    return path


def test_extract_pose_commands_preserves_sparse_times(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "commands.mcap",
        pose_commands=[
            (
                1_000_000_000,
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                [90.0] * 36,
                [20.0] * 36,
            ),
            (
                1_500_000_000,
                4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                [91.0] * 36,
                [21.0] * 36,
            ),
        ],
    )

    rows = extract_pose_commands(
        bag_path=bag,
        episode_index=7,
        episode_start_ns=1_000_000_000,
    )

    assert [row["episode_index"] for row in rows] == [7, 7]
    assert [row["t_ns"] for row in rows] == [1_000_000_000, 1_500_000_000]
    assert rows[0]["time_sec"] == pytest.approx(0.0)
    assert rows[1]["time_sec"] == pytest.approx(0.5)
    assert rows[0]["pose"] == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    assert rows[0]["velocity"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert rows[0]["stiffness"] == [90.0] * 36
    assert rows[0]["damping"] == [20.0] * 36


def test_extract_pose_commands_skips_malformed_decoded_messages(tmp_path: Path):
    bag = _build_mixed_pose_command_mcap(tmp_path / "mixed_commands.mcap")

    rows = extract_pose_commands(
        bag_path=bag,
        episode_index=3,
        episode_start_ns=1_000_000_000,
    )

    assert len(rows) == 1
    assert rows[0]["episode_index"] == 3
    assert rows[0]["t_ns"] == 1_500_000_000
    assert rows[0]["time_sec"] == pytest.approx(0.5)
    assert rows[0]["pose"] == [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
    assert rows[0]["velocity"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_extract_pose_commands_returns_empty_rows_when_topic_missing(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(path=tmp_path / "no_commands.mcap")

    rows = extract_pose_commands(
        bag_path=bag,
        episode_index=3,
        episode_start_ns=1_000_000_000,
    )

    assert rows == []


def test_write_pose_commands_parquet_roundtrip(tmp_path: Path):
    target = tmp_path / "aic" / "pose_commands.parquet"
    rows = [
        {
            "episode_index": 7,
            "t_ns": 1_000_000_000,
            "time_sec": 0.0,
            "pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            "velocity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "stiffness": [90.0] * 36,
            "damping": [20.0] * 36,
        }
    ]

    write_pose_commands_parquet(target, rows)

    loaded = pq.read_table(target).to_pylist()
    assert loaded[0]["episode_index"] == 7
    assert loaded[0]["pose"][0] == pytest.approx(1.0)
    assert len(loaded[0]["stiffness"]) == 36
