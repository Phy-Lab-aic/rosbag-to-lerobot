from pathlib import Path

import pytest
import pyarrow.parquet as pq

from v3_conversion.aic_meta.writer import (
    write_scoring_parquet,
    write_task_parquet,
    write_scene_parquet,
    write_tf_snapshots_parquet,
)


def _task_row(episode_index: int = 0) -> dict:
    return {
        "episode_index": episode_index,
        "run_folder": "run_01_20260412_141241",
        "trial_key": "trial_1",
        "trial_score_folder": "trial_1_score95",
        "schema_version": "0.1.0",
        "cable_type": "sfp_sc", "cable_name": "cable_0",
        "plug_type": "sfp",     "plug_name": "sfp_tip",
        "port_type": "sfp",     "port_name": "sfp_port_0",
        "target_module": "nic_card_mount_0",
        "success": True, "early_terminated": True,
        "early_term_source": "insertion_event",
        "duration_sec": 24.68, "num_steps": 286,
        "policy": "cheatcode", "seed": 42,
        "insertion_event_fired": True,
        "insertion_event_target": "/nic_card_mount_0/sfp_port_0",
        "insertion_event_time_sec": 22.84,
    }


def _scoring_row(episode_index: int = 0) -> dict:
    return {
        "episode_index": episode_index,
        "score_total": 94.68,
        "score_tier1": 1.0, "score_tier2": 18.68, "score_tier3": 75.0,
        "score_contacts": 0.0, "score_contacts_message": "ok",
        "score_duration": 7.63, "score_duration_message": "d",
        "score_insertion_force": 0.0, "score_insertion_force_message": "ok",
        "score_traj_efficiency": 5.88, "score_traj_efficiency_message": "e",
        "score_traj_smoothness": 5.17, "score_traj_smoothness_message": "s",
    }


def test_write_task_parquet_roundtrip(tmp_path: Path):
    target = tmp_path / "aic" / "task.parquet"
    write_task_parquet(target, [_task_row(0), _task_row(1)])
    table = pq.read_table(target)
    assert table.num_rows == 2
    assert table.column("episode_index").to_pylist() == [0, 1]
    assert table.column("insertion_event_target").to_pylist()[0].endswith("sfp_port_0")


def test_write_scoring_parquet_roundtrip(tmp_path: Path):
    target = tmp_path / "aic" / "scoring.parquet"
    write_scoring_parquet(target, [_scoring_row(0)])
    table = pq.read_table(target)
    assert table.num_rows == 1
    import math
    assert math.isclose(table.column("score_total").to_pylist()[0], 94.68, abs_tol=1e-2)


def test_write_scene_parquet_roundtrip(tmp_path: Path):
    target = tmp_path / "aic" / "scene.parquet"
    row = {
        "episode_index": 0,
        "plug_port_distance_init": 0.001,
        "initial_plug_pose_rel_gripper": [0.0, 0.015385, 0.04245,
                                           0.4432, -0.4838, 1.3303],
        "scene_rails": [
            {"name": "nic_rail_0", "entity_present": True, "entity_name": "nic_card_0"},
            {"name": "nic_rail_1", "entity_present": False, "entity_name": ""},
        ],
    }
    write_scene_parquet(target, [row])
    table = pq.read_table(target)
    rails = table.column("scene_rails").to_pylist()[0]
    assert rails[0]["entity_name"] == "nic_card_0"
    assert rails[1]["entity_present"] is False


def test_write_tf_snapshots_parquet_roundtrip(tmp_path: Path):
    target = tmp_path / "aic" / "tf_snapshots.parquet"
    row = {
        "episode_index": 0,
        "scoring_frames_initial": [
            {"frame_id": "task_board", "parent_frame_id": "world",
             "pose": [0.15, -0.20, 1.14, 0.0, 0.0, 0.0, 1.0]},
        ],
        "scoring_frames_final": [
            {"frame_id": "task_board", "parent_frame_id": "world",
             "pose": [0.15, -0.20, 1.14, 0.0, 0.0, 0.0, 1.0]},
        ],
    }
    write_tf_snapshots_parquet(target, [row])
    table = pq.read_table(target)
    first = table.column("scoring_frames_initial").to_pylist()[0][0]
    assert first["frame_id"] == "task_board"
    assert first["pose"][0] == pytest.approx(0.15)
