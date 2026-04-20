"""Extraction config dataclass for the conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Rosbag:
    """Configuration for MCAP frame extraction."""

    topic_map: Dict[str, str] = field(default_factory=dict)
    action_order: List[str] = field(default_factory=list)
    joint_order: Dict[str, Any] = field(default_factory=dict)
    camera_names: List[str] = field(default_factory=list)
    fps: int = 0
    hz_min_ratio: float = 0.7
    robot_type: str = ""
    # Action canonical names whose source topic equals state_topic
    shared_action_names: List[str] = field(default_factory=list)
    wrench_topic: str = ""
