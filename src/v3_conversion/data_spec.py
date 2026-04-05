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

    # --- v2 schema fields ------------------------------------------------

    # Maps canonical name → ROS topic for additional observation sources.
    # e.g. {"wrench": "/fts_broadcaster/wrench",
    #        "controller_state": "/aic_controller/controller_state",
    #        "scoring_tf": "/scoring/tf", "tf_static": "/tf_static"}
    extra_obs_topics: Dict[str, str] = field(default_factory=dict)

    # Maps canonical name → ROS topic for discrete event sources.
    # e.g. {"insertion_event": "/scoring/insertion_event",
    #        "contacts": "/aic/gazebo/contacts/off_limit"}
    event_topics: Dict[str, str] = field(default_factory=dict)

    # Plug tracking parameters for TF-based plug pose extraction.
    # e.g. {"plug_frame": "tip_link"}
    plug_config: Dict[str, Any] = field(default_factory=dict)

    # Path to scoring.yaml for episode-level metadata.
    scoring_yaml_path: str = ""
