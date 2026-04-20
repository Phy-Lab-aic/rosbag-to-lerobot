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
        repo_id="user/ds",
        action_order=["action"],
        joint_order={"obs": ["j0"], "action": {"action": ["j0"]}},
        camera_names=[],
        fps=20,
        root=str(tmp_dataset_root / "out"),
    )


def test_convert_episode_emits_wrench_frames(tmp_dataset_root):
    creator = _make_creator(tmp_dataset_root)
    episode = {
        "obs": np.zeros((2, 1), dtype=np.float32),
        "action": np.zeros((2, 1), dtype=np.float32),
        "images": {},
        "wrench": np.array(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.float32
        ),
        "task": "t",
    }

    creator.convert_episode(episode)

    assert "observation.wrench" in creator.dataset.features
    assert len(creator.dataset.frames) == 2
    assert np.allclose(
        creator.dataset.frames[0]["observation.wrench"], [1, 2, 3, 4, 5, 6]
    )
    assert np.allclose(
        creator.dataset.frames[1]["observation.wrench"], [7, 8, 9, 10, 11, 12]
    )


def test_convert_episode_rejects_wrench_when_dataset_does_not_expect_it(
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
        "obs": np.zeros((2, 1), dtype=np.float32),
        "action": np.zeros((2, 1), dtype=np.float32),
        "images": {},
        "wrench": np.zeros((2, 6), dtype=np.float32),
        "task": "t",
    }

    with pytest.raises(ValueError, match="observation\\.wrench"):
        creator.convert_episode(episode)


def test_convert_episode_rejects_wrench_length_mismatch(tmp_dataset_root):
    creator = _make_creator(tmp_dataset_root)
    creator.dataset = _DummyLeRobotDataset(
        {
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["j0"]},
            "observation.wrench": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            },
        }
    )

    episode = {
        "obs": np.zeros((2, 1), dtype=np.float32),
        "action": np.zeros((2, 1), dtype=np.float32),
        "images": {},
        "wrench": np.zeros((1, 6), dtype=np.float32),
        "task": "t",
    }

    with pytest.raises(ValueError, match="Wrench has 1 frames, expected 2"):
        creator.convert_episode(episode)
