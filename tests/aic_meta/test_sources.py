import json
import textwrap
from pathlib import Path

import pytest

from v3_conversion.aic_meta.sources import (
    load_episode_metadata,
    load_scene_from_config,
    load_run_meta,
    load_scoring_yaml,
    load_tags,
)


def test_load_run_meta_reads_policy_and_seed(tmp_path: Path):
    (tmp_path / "policy.txt").write_text("cheatcode\n")
    (tmp_path / "seed.txt").write_text("42\n")

    result = load_run_meta(tmp_path)

    assert result == {"policy": "cheatcode", "seed": 42}


def test_load_run_meta_missing_files_yields_empties(tmp_path: Path):
    result = load_run_meta(tmp_path)
    assert result == {"policy": "", "seed": -1}


def test_load_tags_picks_relevant_fields(tmp_path: Path):
    (tmp_path / "tags.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "trial": 1,
                "success": True,
                "early_terminated": True,
                "early_term_source": "insertion_event",
            }
        )
    )

    result = load_tags(tmp_path)

    assert result["schema_version"] == "0.1.0"
    assert result["early_term_source"] == "insertion_event"


def test_load_tags_missing_returns_defaults(tmp_path: Path):
    result = load_tags(tmp_path)
    assert result["schema_version"] == ""


def test_load_scoring_yaml_extracts_categories(tmp_path: Path):
    (tmp_path / "scoring.yaml").write_text(
        textwrap.dedent(
            """
            trial_1:
              total: 94.68
              tier_1:
                score: 1.0
                message: Model validation succeeded.
              tier_2:
                score: 18.68
                message: Scoring succeeded.
                categories:
                  contacts: {score: 0.0, message: No contact detected.}
                  duration: {score: 7.63, message: 'Task duration: 25.03 seconds.'}
                  insertion force: {score: 0.0, message: No excessive force detected}
                  trajectory efficiency: {score: 5.88, message: path length info}
                  trajectory smoothness: {score: 5.17, message: jerk info}
              tier_3:
                score: 75.0
                message: Cable insertion successful.
            """
        ).lstrip()
    )

    result = load_scoring_yaml(tmp_path, trial_key="trial_1")

    assert result["score_total"] == pytest.approx(94.68)
    assert result["score_tier1"] == pytest.approx(1.0)
    assert result["score_tier3"] == pytest.approx(75.0)
    assert result["score_contacts"] == pytest.approx(0.0)
    assert result["score_duration_message"] == "Task duration: 25.03 seconds."
    assert result["score_insertion_force_message"] == "No excessive force detected"
    assert result["score_traj_efficiency"] == pytest.approx(5.88)
    assert result["score_traj_smoothness_message"] == "jerk info"


def test_load_episode_metadata_flattens_semantic_and_outcome(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "episode_id": 0,
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module": "nic_card_mount_0",
                "success": True,
                "early_terminated": True,
                "early_term_source": "insertion_event",
                "plug_port_distance": 0.001,
                "num_steps": 286,
                "duration_sec": 24.6827,
            }
        )
    )

    result = load_episode_metadata(tmp_path)

    assert result["cable_type"] == "sfp_sc"
    assert result["target_module"] == "nic_card_mount_0"
    assert result["plug_port_distance_init"] == pytest.approx(0.001)
    assert result["num_steps"] == 286


def test_load_scene_from_config_extracts_rails_and_cable_pose(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        textwrap.dedent(
            """
            trials:
              trial_1:
                scene:
                  task_board:
                    nic_rail_0:
                      entity_present: true
                      entity_name: nic_card_0
                    nic_rail_1:
                      entity_present: false
                    sc_rail_0:
                      entity_present: true
                      entity_name: sc_mount_0
                  cables:
                    cable_0:
                      pose:
                        gripper_offset: {x: 0.0, y: 0.015385, z: 0.04245}
                        roll: 0.4432
                        pitch: -0.4838
                        yaw: 1.3303
            """
        ).lstrip()
    )

    result = load_scene_from_config(tmp_path, trial_key="trial_1")

    assert result["initial_plug_pose_rel_gripper"] == [
        0.0,
        0.015385,
        0.04245,
        0.4432,
        -0.4838,
        1.3303,
    ]
    rail_names = [r["name"] for r in result["scene_rails"]]
    assert rail_names == ["nic_rail_0", "nic_rail_1", "sc_rail_0"]
    assert result["scene_rails"][0]["entity_name"] == "nic_card_0"
    assert result["scene_rails"][1]["entity_present"] is False
    assert result["scene_rails"][1]["entity_name"] == ""
