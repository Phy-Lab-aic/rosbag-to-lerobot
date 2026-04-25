import sys
import types

import numpy as np
import pytest

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


def _make_creator(tmp_dataset_root):
    return DataCreator(
        repo_id="test/labels",
        root=str(tmp_dataset_root),
        robot_type="ur5e",
        action_order=["action"],
        joint_order={"obs": ["j0"], "action": {"action": ["j0"]}},
        camera_names=[],
        fps=20,
    )


def test_data_creator_registers_and_writes_label_features(tmp_dataset_root):
    creator = _make_creator(tmp_dataset_root)
    episode = {
        "obs": np.array([[0.0], [0.2]], dtype=np.float32),
        "action": np.array([[0.2], [0.4]], dtype=np.float32),
        "label.tcp_pose": np.array(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "label.tcp_pose_valid": np.array([True, False], dtype=np.bool_),
        "images": {},
        "task": "Insert cable.",
    }

    creator.convert_episode(episode)

    assert creator.dataset.features["label.tcp_pose"]["shape"] == (7,)
    assert creator.dataset.features["label.tcp_pose_valid"]["dtype"] == "bool"
    assert np.allclose(
        creator.dataset.frames[0]["label.tcp_pose"],
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
    )
    assert creator.dataset.frames[0]["label.tcp_pose_valid"] is True
    assert np.allclose(
        creator.dataset.frames[1]["label.tcp_pose"],
        [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
    )
    assert creator.dataset.frames[1]["label.tcp_pose_valid"] is False


def test_data_creator_rejects_label_length_mismatch(tmp_dataset_root):
    creator = _make_creator(tmp_dataset_root)
    creator.dataset = _DummyLeRobotDataset(
        {
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
            "label.tcp_pose": {"dtype": "float32", "shape": (7,), "names": None},
        }
    )
    episode = {
        "obs": np.array([[0.0], [0.2]], dtype=np.float32),
        "action": np.array([[0.2], [0.4]], dtype=np.float32),
        "label.tcp_pose": np.zeros((1, 7), dtype=np.float32),
        "images": {},
        "task": "Insert cable.",
    }

    with pytest.raises(ValueError, match="label\\.tcp_pose has 1 frames, expected 2"):
        creator.convert_episode(episode)


def test_data_creator_rejects_label_when_dataset_does_not_expect_it(
    tmp_dataset_root,
):
    creator = _make_creator(tmp_dataset_root)
    creator.dataset = _DummyLeRobotDataset(
        {
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
        }
    )
    episode = {
        "obs": np.array([[0.0], [0.2]], dtype=np.float32),
        "action": np.array([[0.2], [0.4]], dtype=np.float32),
        "label.tcp_pose": np.zeros((2, 7), dtype=np.float32),
        "images": {},
        "task": "Insert cable.",
    }

    with pytest.raises(ValueError, match="label\\.tcp_pose"):
        creator.convert_episode(episode)


def test_data_creator_rejects_missing_label_when_dataset_expects_it(
    tmp_dataset_root,
):
    creator = _make_creator(tmp_dataset_root)
    creator.dataset = _DummyLeRobotDataset(
        {
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
            "label.tcp_pose": {"dtype": "float32", "shape": (7,), "names": None},
        }
    )
    episode = {
        "obs": np.array([[0.0], [0.2]], dtype=np.float32),
        "action": np.array([[0.2], [0.4]], dtype=np.float32),
        "images": {},
        "task": "Insert cable.",
    }

    with pytest.raises(ValueError, match="label features missing"):
        creator.convert_episode(episode)
