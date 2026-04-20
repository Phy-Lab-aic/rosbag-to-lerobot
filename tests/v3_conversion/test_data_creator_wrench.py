import sys
import types

import numpy as np

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

    @classmethod
    def create(cls, **kwargs):
        return cls(kwargs["features"])


lerobot_dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
lerobot_dataset_mod.LeRobotDataset = _DummyLeRobotDataset
sys.modules["lerobot.datasets.lerobot_dataset"] = lerobot_dataset_mod
setattr(datasets_pkg, "lerobot_dataset", lerobot_dataset_mod)

from v3_conversion.data_creator import DataCreator


def test_create_dataset_registers_wrench_feature(tmp_dataset_root):
    creator = DataCreator(
        repo_id="user/ds",
        action_order=["action"],
        joint_order={"obs": ["j0"], "action": {"action": ["j0"]}},
        camera_names=[],
        fps=20,
        root=str(tmp_dataset_root / "out"),
    )
    episode = {
        "obs": np.zeros((2, 1), dtype=np.float32),
        "action": np.zeros((2, 1), dtype=np.float32),
        "images": {},
        "wrench": np.zeros((2, 6), dtype=np.float32),
        "task": "t",
    }
    creator.create_dataset(episode)
    assert "observation.wrench" in creator.dataset.features
    assert tuple(creator.dataset.features["observation.wrench"]["shape"]) == (6,)
