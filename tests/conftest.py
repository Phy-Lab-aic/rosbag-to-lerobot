"""Shared pytest fixtures for the rosbag-to-lerobot test suite."""

from pathlib import Path
from typing import Any, Dict

import pytest


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

