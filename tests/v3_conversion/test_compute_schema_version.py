import sys
import types

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

# Mirror the stub shape used by neighbouring data_creator tests so that
# `data_creator` binds to a LeRobotDataset class with the attributes other
# tests rely on. Avoids pollution when this test loads first alphabetically.
class _DummyLeRobotDataset:
    def __init__(self, features=None):
        self.features = features or {}
        self.episode_buffer = None
        self.frames = []
        self.saved_episodes = 0

    @classmethod
    def create(cls, **kwargs):
        return cls(kwargs.get("features"))

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


from v3_conversion.data_creator import (  # noqa: E402  (import after stubs)
    DATASET_SCHEMA_VERSIONS,
    compute_schema_version,
)


_BASE_FEATURES = {
    "observation.state": {},
    "action": {},
    "observation.images.cam_center": {},
}


@pytest.mark.parametrize(
    "extras,expected",
    [
        ((), "0.0.0"),
        (("observation.wrench",), "0.1.0"),
        (("observation.velocity",), "0.2.0"),
        (("observation.velocity", "observation.wrench"), "0.3.0"),
    ],
)
def test_compute_schema_version_known_shapes(extras, expected):
    features = {**_BASE_FEATURES, **{k: {} for k in extras}}
    assert compute_schema_version(features) == expected


def test_compute_schema_version_table_is_consistent():
    # Sanity-check the lookup table: every value must be unique so two shapes
    # never share a label.
    versions = list(DATASET_SCHEMA_VERSIONS.values())
    assert len(versions) == len(set(versions))


def test_compute_schema_version_unknown_shape_falls_back(caplog, monkeypatch):
    # Simulate an expansion of optional features without updating the version
    # table — the function should warn and return the experimental sentinel.
    from v3_conversion import data_creator as dc_module

    extended_optional = dc_module._OPTIONAL_DATASET_FEATURES | {
        "observation.touch"
    }
    monkeypatch.setattr(
        dc_module, "_OPTIONAL_DATASET_FEATURES", extended_optional
    )
    features = {**_BASE_FEATURES, "observation.touch": {}}
    with caplog.at_level("WARNING"):
        version = compute_schema_version(features)
    assert version == "0.0.0-experimental"
