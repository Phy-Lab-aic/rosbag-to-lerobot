from types import SimpleNamespace

import numpy as np

from v3_conversion.data_converter import build_frame, frames_to_episode


def _wrench_msg():
    return SimpleNamespace(
        header=SimpleNamespace(),
        wrench=SimpleNamespace(
            force=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            torque=SimpleNamespace(x=0.4, y=0.5, z=0.6),
        ),
    )


def _joint_msg(positions):
    return SimpleNamespace(
        header=SimpleNamespace(),
        name=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "gripper/left_finger_joint",
        ],
        position=list(positions),
        velocity=[0.0] * 7,
        effort=[0.0] * 7,
    )


def test_build_frame_emits_wrench_key():
    joint_order = {
        "obs": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "gripper/left_finger_joint",
        ],
        "action": {
            "action": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "gripper/left_finger_joint",
            ]
        },
    }
    frame = build_frame(
        image_msgs={},
        follower_msgs={"observation": _joint_msg([0.1] * 7)},
        leader_msgs={"action": _joint_msg([0.2] * 7)},
        joint_order=joint_order,
        rot_img=False,
        schema_map={
            "observation": "sensor_msgs/msg/JointState",
            "action": "sensor_msgs/msg/JointState",
            "wrench": "geometry_msgs/msg/WrenchStamped",
        },
        wrench_msg=_wrench_msg(),
    )
    assert frame is not None
    assert np.allclose(frame["wrench"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_frames_to_episode_stacks_wrench():
    joint_order = {
        "obs": ["j0"],
        "action": {"action": ["j0"]},
    }
    frames = [
        {
            "images": {},
            "obs": np.array([0.0], dtype=np.float32),
            "action": {"action": np.array([0.1], dtype=np.float32)},
            "wrench": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        },
        {
            "images": {},
            "obs": np.array([0.2], dtype=np.float32),
            "action": {"action": np.array([0.3], dtype=np.float32)},
            "wrench": np.array([7, 8, 9, 10, 11, 12], dtype=np.float32),
        },
    ]
    episode = frames_to_episode(
        frames,
        action_order=["action"],
        camera_names=[],
        task="t",
    )
    assert episode["wrench"].shape == (2, 6)
    assert np.allclose(episode["wrench"][1], [7, 8, 9, 10, 11, 12])
