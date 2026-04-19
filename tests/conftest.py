"""Shared pytest fixtures for the rosbag-to-lerobot test suite."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from mcap_ros2.writer import Writer as Ros2Writer


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
            wrench=[(t_ns, fx,fy,fz, tx,ty,tz), ...],
            images={"cam_left": [(t_ns, h, w, bytes)]},
            insertion_event=[(t_ns, "/nic_card_mount_0/sfp_port_0")],
            scoring_tf=[(t_ns, [(parent, child, x,y,z, qx,qy,qz,qw)])],
        )
    """

    def _build(
        path: Path = tmp_path / "sample.mcap",
        *,
        joint_states=None,
        wrench=None,
        images=None,
        insertion_event=None,
        scoring_tf=None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            writer = Ros2Writer(f)
            string_schema = None

            if joint_states:
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
                for t_ns, names, positions in joint_states:
                    writer.write_message(
                        topic="/joint_states",
                        schema=joint_state_schema,
                        message={
                            "header": {
                                "stamp": {
                                    "sec": t_ns // 1_000_000_000,
                                    "nanosec": t_ns % 1_000_000_000,
                                },
                                "frame_id": "base_link",
                            },
                            "name": list(names),
                            "position": list(positions),
                            "velocity": [0.0] * len(names),
                            "effort": [0.0] * len(names),
                        },
                        log_time=t_ns,
                        publish_time=t_ns,
                    )
            if insertion_event:
                string_schema = string_schema or writer.register_msgdef(
                    datatype="std_msgs/msg/String",
                    msgdef_text="string data",
                )
                for t_ns, data in insertion_event:
                    writer.write_message(
                        topic="/scoring/insertion_event",
                        schema=string_schema,
                        message={"data": data},
                        log_time=t_ns,
                        publish_time=t_ns,
                    )
            # wrench, images, insertion_event, scoring_tf similarly — expand in tasks that need them
            writer.finish()
        return path

    return _build
