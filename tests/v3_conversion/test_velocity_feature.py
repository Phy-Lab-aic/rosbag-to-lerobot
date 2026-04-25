import sys
import types
from types import SimpleNamespace

import numpy as np

from v3_conversion.data_converter import build_frame, frames_to_episode

sys.modules.setdefault("av", types.ModuleType("av"))

lerobot_pkg = sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
datasets_pkg = sys.modules.setdefault(
    "lerobot.datasets", types.ModuleType("lerobot.datasets")
)
setattr(lerobot_pkg, "datasets", datasets_pkg)

dataset_metadata_mod = types.ModuleType("lerobot.datasets.dataset_metadata")
dataset_metadata_mod.CODEBASE_VERSION = "v3.0.0"
sys.modules["lerobot.datasets.dataset_metadata"] = dataset_metadata_mod
setattr(datasets_pkg, "dataset_metadata", dataset_metadata_mod)


class _DummyLeRobotDataset:
    def __init__(self, features):
        self.features = features
        self.episode_buffer = None
        self.frames = []
        self.saved_episodes = 0

    @classmethod
    def create(cls, **kwargs):
        return cls(kwargs["features"])

    def create_episode_buffer(self):
        return {"size": 0}

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


lerobot_dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
lerobot_dataset_mod.LeRobotDataset = _DummyLeRobotDataset
sys.modules["lerobot.datasets.lerobot_dataset"] = lerobot_dataset_mod
setattr(datasets_pkg, "lerobot_dataset", lerobot_dataset_mod)

from v3_conversion.data_creator import DataCreator


JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper/left_finger_joint",
]


def _joint_msg(names, positions, velocities):
    return SimpleNamespace(
        name=list(names),
        position=list(positions),
        velocity=list(velocities),
        effort=[0.0] * len(names),
    )


def test_build_frame_extracts_velocity_in_canonical_order():
    msg = _joint_msg(
        ["elbow_joint", "shoulder_pan_joint", "shoulder_lift_joint"],
        [0.3, 0.1, 0.2],
        [3.0, 1.0, 2.0],
    )

    frame = build_frame(
        image_msgs={},
        follower_msgs={"observation": msg},
        leader_msgs={"action": msg},
        joint_order={
            "obs": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
            "action": {
                "action": [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                ]
            },
        },
        rot_img=False,
        schema_map={
            "observation": "sensor_msgs/msg/JointState",
            "action": "sensor_msgs/msg/JointState",
        },
    )

    assert np.allclose(frame["obs"], [0.1, 0.2, 0.3])
    assert np.allclose(frame["velocity"], [1.0, 2.0, 3.0])


def test_frames_to_episode_stacks_velocity():
    frames = [
        {
            "images": {},
            "obs": np.array([0.0], dtype=np.float32),
            "velocity": np.array([1.0], dtype=np.float32),
            "action": {"action": np.array([0.1], dtype=np.float32)},
        },
        {
            "images": {},
            "obs": np.array([0.2], dtype=np.float32),
            "velocity": np.array([2.0], dtype=np.float32),
            "action": {"action": np.array([0.3], dtype=np.float32)},
        },
    ]

    episode = frames_to_episode(
        frames,
        action_order=["action"],
        camera_names=[],
        task="t",
    )

    assert np.allclose(episode["velocity"], [[1.0], [2.0]])


def test_data_creator_registers_and_writes_observation_velocity(tmp_dataset_root):
    creator = DataCreator(
        repo_id="test/velocity",
        root=str(tmp_dataset_root),
        robot_type="ur5e",
        action_order=["action"],
        joint_order={"obs": ["j0"], "action": {"action": ["j0"]}},
        camera_names=[],
        fps=20,
    )
    episode = {
        "obs": np.array([[0.0], [0.2]], dtype=np.float32),
        "velocity": np.array([[1.0], [2.0]], dtype=np.float32),
        "action": np.array([[0.2], [0.4]], dtype=np.float32),
        "images": {},
        "task": "Insert cable.",
    }

    creator.convert_episode(episode)

    assert "observation.velocity" in creator.dataset.features
    assert np.allclose(creator.dataset.frames[0]["observation.velocity"], [1.0])
