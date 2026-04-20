import importlib
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import pyarrow.parquet as pq

from v3_conversion.aic_meta.writer import (
    write_scene_parquet,
    write_scoring_parquet,
    write_task_parquet,
    write_tf_snapshots_parquet,
)


def _import_main(monkeypatch):
    sys.modules.pop("main", None)

    data_creator_mod = types.ModuleType("v3_conversion.data_creator")

    class _PlaceholderDataCreator:
        pass

    data_creator_mod.DataCreator = _PlaceholderDataCreator
    monkeypatch.setitem(sys.modules, "v3_conversion.data_creator", data_creator_mod)

    lerobot_pkg = types.ModuleType("lerobot")
    datasets_pkg = types.ModuleType("lerobot.datasets")
    lerobot_dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")

    class _RecordingLeRobotDataset:
        push_calls = []

        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = root

        def push_to_hub(self, **kwargs):
            type(self).push_calls.append(
                {
                    "repo_id": self.repo_id,
                    "root": self.root,
                    "kwargs": kwargs,
                }
            )

    lerobot_dataset_mod.LeRobotDataset = _RecordingLeRobotDataset
    datasets_pkg.lerobot_dataset = lerobot_dataset_mod
    lerobot_pkg.datasets = datasets_pkg

    monkeypatch.setitem(sys.modules, "lerobot", lerobot_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_pkg)
    monkeypatch.setitem(
        sys.modules, "lerobot.datasets.lerobot_dataset", lerobot_dataset_mod
    )

    module = importlib.import_module("main")
    return importlib.reload(module), _RecordingLeRobotDataset


def _make_input_tree(tmp_path: Path) -> tuple[Path, Path]:
    input_root = tmp_path / "input"
    run_dir = input_root / "run_001"
    (run_dir / "trial_1_score95" / "episode").mkdir(parents=True)
    (run_dir / "run_001_0.mcap").write_bytes(b"mcap")
    return input_root, run_dir


def _fake_config(repo_id: str = "aic_task") -> dict:
    return {
        "task_name": "aic_task",
        "repo_id": repo_id,
        "folders": ["run_001"],
        "robot_type": "ur5e",
        "fps": 20,
        "camera_topic_map": {"cam_left": "/left_camera/image"},
        "joint_names": ["joint_1"],
        "state_topic": "/joint_states",
        "wrench_topic": "/fts_broadcaster/wrench",
        "action_topics_map": {"leader": "/joint_states"},
        "task_instruction": [],
        "tags": [],
    }


def _fake_bag_config():
    return SimpleNamespace(
        robot_type="ur5e",
        action_order=["action"],
        joint_order={"obs": ["joint_1"], "action": {"action": ["joint_1"]}},
        camera_names=["cam_left"],
        fps=20,
        hz_min_ratio=0.7,
        topic_map={"/left_camera/image": "cam_left"},
    )


def _task_row(episode_index: int) -> dict:
    return {
        "episode_index": episode_index,
        "run_folder": f"run_{episode_index:03d}",
        "trial_key": "trial_1",
        "trial_score_folder": "trial_1_score95",
        "schema_version": "0.1.0",
        "cable_type": "sfp_sc",
        "cable_name": "cable_0",
        "plug_type": "sfp",
        "plug_name": "sfp_tip",
        "port_type": "sfp",
        "port_name": "sfp_port_0",
        "target_module": "nic_card_mount_0",
        "success": True,
        "early_terminated": False,
        "early_term_source": "",
        "duration_sec": 24.68,
        "num_steps": 286,
        "policy": "cheatcode",
        "seed": 42,
        "insertion_event_fired": True,
        "insertion_event_target": "/nic_card_mount_0/sfp_port_0",
        "insertion_event_time_sec": 22.84,
    }


def _scoring_row(episode_index: int) -> dict:
    return {
        "episode_index": episode_index,
        "score_total": 94.68,
        "score_tier1": 1.0,
        "score_tier2": 18.68,
        "score_tier3": 75.0,
        "score_contacts": 0.0,
        "score_contacts_message": "ok",
        "score_duration": 7.63,
        "score_duration_message": "d",
        "score_insertion_force": 0.0,
        "score_insertion_force_message": "ok",
        "score_traj_efficiency": 5.88,
        "score_traj_efficiency_message": "e",
        "score_traj_smoothness": 5.17,
        "score_traj_smoothness_message": "s",
    }


def _scene_row(episode_index: int) -> dict:
    return {
        "episode_index": episode_index,
        "plug_port_distance_init": 0.001,
        "initial_plug_pose_rel_gripper": [0.0, 0.015385, 0.04245, 0.4432, -0.4838, 1.3303],
        "scene_rails": [
            {
                "name": "nic_rail_0",
                "entity_present": True,
                "entity_name": "nic_card_0",
            }
        ],
    }


def _tf_row(episode_index: int) -> dict:
    return {
        "episode_index": episode_index,
        "scoring_frames_initial": [
            {
                "frame_id": "task_board",
                "parent_frame_id": "world",
                "pose": [0.15, -0.20, 1.14, 0.0, 0.0, 0.0, 1.0],
            }
        ],
        "scoring_frames_final": [
            {
                "frame_id": "task_board",
                "parent_frame_id": "world",
                "pose": [0.15, -0.20, 1.14, 0.0, 0.0, 0.0, 1.0],
            }
        ],
    }


def test_run_conversion_does_not_persist_episode_when_aic_meta_extraction_fails(
    monkeypatch, tmp_path
):
    main, _ = _import_main(monkeypatch)
    input_root, run_dir = _make_input_tree(tmp_path)

    class FakeDataCreator:
        instances = []

        def __init__(self, **kwargs):
            self.dataset = None
            self.convert_calls = 0
            FakeDataCreator.instances.append(self)

        def convert_episode(self, episode):
            self.convert_calls += 1
            self.dataset = SimpleNamespace(
                meta=SimpleNamespace(total_episodes=1),
                finalize=lambda: None,
            )

        def recover_dataset_state(self):
            return None

        def correct_video_timestamps(self):
            return None

        def patch_episodes_metadata(self):
            return None

    monkeypatch.setattr(main, "INPUT_PATH", input_root)
    monkeypatch.setattr(main, "OUTPUT_PATH", tmp_path / "output")
    monkeypatch.setattr(main, "DataCreator", FakeDataCreator)
    monkeypatch.setattr(main, "_load_config", lambda _: _fake_config())
    monkeypatch.setattr(main, "_load_metacard", lambda folder, defaults=None: {})
    monkeypatch.setattr(
        main,
        "_prepare_config",
        lambda folder, metadata, robot_type, fps: (run_dir / "run_001_0.mcap", _fake_bag_config()),
    )
    monkeypatch.setattr(
        main, "validate_mcap_topics", lambda bag_path, topic_map: {"missing_topics": []}
    )
    monkeypatch.setattr(
        main,
        "extract_frames",
        lambda bag_path, config: (
            [{"obs": [1.0], "action": {"action": [2.0]}, "images": {}, "emitted_timestamp_ns": 50}],
            {"cam_left": [0, 50]},
        ),
    )
    monkeypatch.setattr(
        main,
        "validate_from_timestamps",
        lambda **kwargs: SimpleNamespace(is_valid=True, overall_message="ok"),
    )
    monkeypatch.setattr(
        main,
        "load_episode_metadata",
        lambda episode_dir: {
            "cable_type": "cable",
            "cable_name": "cable_0",
            "plug_type": "plug",
            "plug_name": "plug_0",
            "port_type": "port",
            "port_name": "port_0",
            "target_module": "module",
            "success": True,
            "early_terminated": False,
            "early_term_source": "",
            "duration_sec": 1.0,
            "num_steps": 1,
            "plug_port_distance_init": 0.1,
        },
    )
    monkeypatch.setattr(main, "build_task_string", lambda episode_meta: "task")
    monkeypatch.setattr(
        main,
        "frames_to_episode",
        lambda frames, action_order, camera_names, task: {
            "obs": [[1.0]],
            "action": [[2.0]],
            "images": {},
            "task": task,
        },
    )
    monkeypatch.setattr(main, "apply_one_step_shift", lambda episode: episode)
    monkeypatch.setattr(main, "load_run_meta", lambda run_dir: {"policy": "pi0", "seed": 7})
    monkeypatch.setattr(main, "load_tags", lambda trial_dir: (_ for _ in ()).throw(RuntimeError("tags failed")))
    monkeypatch.setattr(main, "load_scoring_yaml", lambda trial_dir, trial_key: {"score_total": 1.0})
    monkeypatch.setattr(
        main,
        "load_scene_from_config",
        lambda run_dir, trial_key: {
            "initial_plug_pose_rel_gripper": [0.0] * 6,
            "scene_rails": [],
        },
    )
    monkeypatch.setattr(
        main,
        "extract_insertion_event",
        lambda bag_path, episode_start_ns: {
            "insertion_event_fired": False,
            "insertion_event_target": "",
            "insertion_event_time_sec": float("nan"),
        },
    )
    monkeypatch.setattr(
        main,
        "extract_scoring_tf_snapshots",
        lambda bag_path: {"scoring_frames_initial": [], "scoring_frames_final": []},
    )

    result = main.run_conversion("ignored.json")

    assert result == 1
    assert len(FakeDataCreator.instances) == 1
    assert FakeDataCreator.instances[0].convert_calls == 0


def test_run_conversion_uses_first_emitted_frame_timestamp_for_insertion_event(
    monkeypatch, tmp_path
):
    main, _ = _import_main(monkeypatch)
    input_root, run_dir = _make_input_tree(tmp_path)
    captured = {}

    class FakeDataset:
        def __init__(self):
            self.meta = SimpleNamespace(total_episodes=0)

        def finalize(self):
            return None

    class FakeDataCreator:
        def __init__(self, **kwargs):
            self.dataset = None

        def convert_episode(self, episode):
            if self.dataset is None:
                self.dataset = FakeDataset()
            self.dataset.meta.total_episodes += 1

        def recover_dataset_state(self):
            return None

        def correct_video_timestamps(self):
            return None

        def patch_episodes_metadata(self):
            return None

    monkeypatch.setattr(main, "INPUT_PATH", input_root)
    monkeypatch.setattr(main, "OUTPUT_PATH", tmp_path / "output")
    monkeypatch.setattr(main, "DataCreator", FakeDataCreator)
    monkeypatch.setattr(main, "_load_config", lambda _: _fake_config())
    monkeypatch.setattr(main, "_load_metacard", lambda folder, defaults=None: {})
    monkeypatch.setattr(
        main,
        "_prepare_config",
        lambda folder, metadata, robot_type, fps: (run_dir / "run_001_0.mcap", _fake_bag_config()),
    )
    monkeypatch.setattr(
        main, "validate_mcap_topics", lambda bag_path, topic_map: {"missing_topics": []}
    )
    monkeypatch.setattr(
        main,
        "extract_frames",
        lambda bag_path, config: (
            [{"obs": [1.0], "action": {"action": [2.0]}, "images": {}, "emitted_timestamp_ns": 50_000_000}],
            {"cam_left": [0, 50_000_000]},
        ),
    )
    monkeypatch.setattr(
        main,
        "validate_from_timestamps",
        lambda **kwargs: SimpleNamespace(is_valid=True, overall_message="ok"),
    )
    monkeypatch.setattr(
        main,
        "load_episode_metadata",
        lambda episode_dir: {
            "cable_type": "cable",
            "cable_name": "cable_0",
            "plug_type": "plug",
            "plug_name": "plug_0",
            "port_type": "port",
            "port_name": "port_0",
            "target_module": "module",
            "success": True,
            "early_terminated": False,
            "early_term_source": "",
            "duration_sec": 1.0,
            "num_steps": 1,
            "plug_port_distance_init": 0.1,
        },
    )
    monkeypatch.setattr(main, "build_task_string", lambda episode_meta: "task")
    monkeypatch.setattr(
        main,
        "frames_to_episode",
        lambda frames, action_order, camera_names, task: {
            "obs": [[1.0]],
            "action": [[2.0]],
            "images": {},
            "task": task,
        },
    )
    monkeypatch.setattr(main, "apply_one_step_shift", lambda episode: episode)
    monkeypatch.setattr(main, "load_run_meta", lambda run_dir: {"policy": "pi0", "seed": 7})
    monkeypatch.setattr(main, "load_tags", lambda trial_dir: {"schema_version": "1.0"})
    monkeypatch.setattr(main, "load_scoring_yaml", lambda trial_dir, trial_key: {"score_total": 1.0})
    monkeypatch.setattr(
        main,
        "load_scene_from_config",
        lambda run_dir, trial_key: {
            "initial_plug_pose_rel_gripper": [0.0] * 6,
            "scene_rails": [],
        },
    )

    def fake_extract_insertion_event(bag_path, episode_start_ns):
        captured["episode_start_ns"] = episode_start_ns
        return {
            "insertion_event_fired": True,
            "insertion_event_target": "target",
            "insertion_event_time_sec": 0.2,
        }

    monkeypatch.setattr(main, "extract_insertion_event", fake_extract_insertion_event)
    monkeypatch.setattr(
        main,
        "extract_scoring_tf_snapshots",
        lambda bag_path: {"scoring_frames_initial": [], "scoring_frames_final": []},
    )
    monkeypatch.setattr(main, "write_task_parquet", lambda target, rows: None)
    monkeypatch.setattr(main, "write_scoring_parquet", lambda target, rows: None)
    monkeypatch.setattr(main, "write_scene_parquet", lambda target, rows: None)
    monkeypatch.setattr(main, "write_tf_snapshots_parquet", lambda target, rows: None)

    result = main.run_conversion("ignored.json")

    assert result == 0
    assert captured["episode_start_ns"] == 50_000_000


def test_run_conversion_skips_push_to_hub_when_finalize_metadata_write_fails(
    monkeypatch, tmp_path, capsys
):
    main, recording_dataset_cls = _import_main(monkeypatch)
    input_root, run_dir = _make_input_tree(tmp_path)

    class FakeDataset:
        def __init__(self):
            self.meta = SimpleNamespace(total_episodes=0)

        def finalize(self):
            return None

    class FakeDataCreator:
        def __init__(self, **kwargs):
            self.dataset = None

        def convert_episode(self, episode):
            if self.dataset is None:
                self.dataset = FakeDataset()
            self.dataset.meta.total_episodes += 1

        def recover_dataset_state(self):
            return None

        def correct_video_timestamps(self):
            return None

        def patch_episodes_metadata(self):
            return None

    monkeypatch.setattr(main, "INPUT_PATH", input_root)
    monkeypatch.setattr(main, "OUTPUT_PATH", tmp_path / "output")
    monkeypatch.setattr(main, "DataCreator", FakeDataCreator)
    monkeypatch.setattr(main, "_load_config", lambda _: _fake_config(repo_id="org/dataset"))
    monkeypatch.setattr(main, "_load_metacard", lambda folder, defaults=None: {})
    monkeypatch.setattr(
        main,
        "_prepare_config",
        lambda folder, metadata, robot_type, fps: (run_dir / "run_001_0.mcap", _fake_bag_config()),
    )
    monkeypatch.setattr(
        main, "validate_mcap_topics", lambda bag_path, topic_map: {"missing_topics": []}
    )
    monkeypatch.setattr(
        main,
        "extract_frames",
        lambda bag_path, config: (
            [{"obs": [1.0], "action": {"action": [2.0]}, "images": {}, "emitted_timestamp_ns": 50_000_000}],
            {"cam_left": [50_000_000]},
        ),
    )
    monkeypatch.setattr(
        main,
        "validate_from_timestamps",
        lambda **kwargs: SimpleNamespace(is_valid=True, overall_message="ok"),
    )
    monkeypatch.setattr(
        main,
        "load_episode_metadata",
        lambda episode_dir: {
            "cable_type": "cable",
            "cable_name": "cable_0",
            "plug_type": "plug",
            "plug_name": "plug_0",
            "port_type": "port",
            "port_name": "port_0",
            "target_module": "module",
            "success": True,
            "early_terminated": False,
            "early_term_source": "",
            "duration_sec": 1.0,
            "num_steps": 1,
            "plug_port_distance_init": 0.1,
        },
    )
    monkeypatch.setattr(main, "build_task_string", lambda episode_meta: "task")
    monkeypatch.setattr(
        main,
        "frames_to_episode",
        lambda frames, action_order, camera_names, task: {
            "obs": [[1.0]],
            "action": [[2.0]],
            "images": {},
            "task": task,
        },
    )
    monkeypatch.setattr(main, "apply_one_step_shift", lambda episode: episode)
    monkeypatch.setattr(main, "load_run_meta", lambda run_dir: {"policy": "pi0", "seed": 7})
    monkeypatch.setattr(main, "load_tags", lambda trial_dir: {"schema_version": "1.0"})
    monkeypatch.setattr(main, "load_scoring_yaml", lambda trial_dir, trial_key: {"score_total": 1.0})
    monkeypatch.setattr(
        main,
        "load_scene_from_config",
        lambda run_dir, trial_key: {
            "initial_plug_pose_rel_gripper": [0.0] * 6,
            "scene_rails": [],
        },
    )
    monkeypatch.setattr(
        main,
        "extract_insertion_event",
        lambda bag_path, episode_start_ns: {
            "insertion_event_fired": True,
            "insertion_event_target": "target",
            "insertion_event_time_sec": 0.2,
        },
    )
    monkeypatch.setattr(
        main,
        "extract_scoring_tf_snapshots",
        lambda bag_path: {"scoring_frames_initial": [], "scoring_frames_final": []},
    )
    monkeypatch.setattr(
        main,
        "write_task_parquet",
        lambda target, rows: (_ for _ in ()).throw(RuntimeError("write failed")),
    )

    result = main.run_conversion("ignored.json")
    captured = capsys.readouterr()

    assert result == 2
    assert recording_dataset_cls.push_calls == []
    assert "Failed to finalize dataset or write AIC metadata" in captured.err
    assert "Skipping push_to_hub because dataset finalization or AIC metadata write failed" in captured.err


def test_run_conversion_merge_mode_preserves_existing_aic_rows(
    monkeypatch, tmp_path
):
    main, _ = _import_main(monkeypatch)
    input_root, run_dir = _make_input_tree(tmp_path)
    existing_root = tmp_path / "existing_dataset"
    aic_dir = existing_root / "meta" / "aic"
    aic_dir.mkdir(parents=True, exist_ok=True)
    (existing_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "fps": 20,
                "features": {
                    "observation.state": {"shape": [1]},
                    "action": {"shape": [1]},
                },
            }
        )
    )
    write_task_parquet(aic_dir / "task.parquet", [_task_row(0)])
    write_scoring_parquet(aic_dir / "scoring.parquet", [_scoring_row(0)])
    write_scene_parquet(aic_dir / "scene.parquet", [_scene_row(0)])
    write_tf_snapshots_parquet(aic_dir / "tf_snapshots.parquet", [_tf_row(0)])

    class FakeDataset:
        def __init__(self, total_episodes=1):
            self.meta = SimpleNamespace(total_episodes=total_episodes)

        def finalize(self):
            return None

    class FakeDataCreator:
        def __init__(self, **kwargs):
            self.dataset = None

        def convert_episode(self, episode):
            if self.dataset is None:
                self.dataset = FakeDataset()
            self.dataset.meta.total_episodes += 1

        def recover_dataset_state(self):
            return None

        def correct_video_timestamps(self):
            return None

        def patch_episodes_metadata(self):
            return None

    monkeypatch.setattr(main, "INPUT_PATH", input_root)
    monkeypatch.setattr(main, "OUTPUT_PATH", tmp_path / "unused_output")
    monkeypatch.setattr(main, "DataCreator", FakeDataCreator)
    monkeypatch.setattr(main, "_load_config", lambda _: _fake_config())
    monkeypatch.setattr(main, "_load_metacard", lambda folder, defaults=None: {})
    monkeypatch.setattr(
        main,
        "_prepare_config",
        lambda folder, metadata, robot_type, fps: (run_dir / "run_001_0.mcap", _fake_bag_config()),
    )
    monkeypatch.setattr(
        main, "validate_mcap_topics", lambda bag_path, topic_map: {"missing_topics": []}
    )
    monkeypatch.setattr(
        main,
        "extract_frames",
        lambda bag_path, config: (
            [{"obs": [1.0], "action": {"action": [2.0]}, "images": {}, "emitted_timestamp_ns": 50_000_000}],
            {"cam_left": [50_000_000]},
        ),
    )
    monkeypatch.setattr(
        main,
        "validate_from_timestamps",
        lambda **kwargs: SimpleNamespace(is_valid=True, overall_message="ok"),
    )
    monkeypatch.setattr(
        main,
        "load_episode_metadata",
        lambda episode_dir: {
            "cable_type": "cable",
            "cable_name": "cable_0",
            "plug_type": "plug",
            "plug_name": "plug_0",
            "port_type": "port",
            "port_name": "port_0",
            "target_module": "module",
            "success": True,
            "early_terminated": False,
            "early_term_source": "",
            "duration_sec": 1.0,
            "num_steps": 1,
            "plug_port_distance_init": 0.1,
        },
    )
    monkeypatch.setattr(main, "build_task_string", lambda episode_meta: "task")
    monkeypatch.setattr(
        main,
        "frames_to_episode",
        lambda frames, action_order, camera_names, task: {
            "obs": [[1.0]],
            "action": [[2.0]],
            "images": {},
            "task": task,
        },
    )
    monkeypatch.setattr(main, "apply_one_step_shift", lambda episode: episode)
    monkeypatch.setattr(main, "load_run_meta", lambda run_dir: {"policy": "pi0", "seed": 7})
    monkeypatch.setattr(main, "load_tags", lambda trial_dir: {"schema_version": "1.0"})
    monkeypatch.setattr(main, "load_scoring_yaml", lambda trial_dir, trial_key: {"score_total": 1.0})
    monkeypatch.setattr(
        main,
        "load_scene_from_config",
        lambda run_dir, trial_key: {
            "initial_plug_pose_rel_gripper": [0.0] * 6,
            "scene_rails": [],
        },
    )
    monkeypatch.setattr(
        main,
        "extract_insertion_event",
        lambda bag_path, episode_start_ns: {
            "insertion_event_fired": True,
            "insertion_event_target": "target",
            "insertion_event_time_sec": 0.2,
        },
    )
    monkeypatch.setattr(
        main,
        "extract_scoring_tf_snapshots",
        lambda bag_path: {"scoring_frames_initial": [], "scoring_frames_final": []},
    )

    result = main.run_conversion(
        "ignored.json", output_dir=str(existing_root), merge=True
    )

    assert result == 0
    assert pq.read_table(aic_dir / "task.parquet").column("episode_index").to_pylist() == [0, 1]
    assert pq.read_table(aic_dir / "scoring.parquet").column("episode_index").to_pylist() == [0, 1]
    assert pq.read_table(aic_dir / "scene.parquet").column("episode_index").to_pylist() == [0, 1]
    assert pq.read_table(aic_dir / "tf_snapshots.parquet").column("episode_index").to_pylist() == [0, 1]


def test_load_config_and_metacard_propagate_wrench_topic(monkeypatch, tmp_path):
    main, _ = _import_main(monkeypatch)

    config_path = tmp_path / "config_merge.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "aic_task",
                "repo_id": "Phy-lab/basic_aic_cheetcode_dataset",
                "robot": "ur5e",
                "fps": 20,
                "folders": ["run_001"],
                "camera_topic_map": {"cam_left": "/left_camera/image"},
                "joint_names": ["joint_1"],
                "state_topic": "/joint_states",
                "wrench_topic": "/fts_broadcaster/wrench",
                "action_topics_map": {"leader": "/joint_states"},
                "task_instruction": [],
                "tags": [],
            }
        )
    )

    input_root, _ = _make_input_tree(tmp_path)
    monkeypatch.setattr(main, "INPUT_PATH", input_root)

    cfg = main._load_config(str(config_path))
    metacard = main._load_metacard("run_001", {"wrench_topic": cfg["wrench_topic"]})

    assert cfg["wrench_topic"] == "/fts_broadcaster/wrench"
    assert metacard["wrench_topic"] == "/fts_broadcaster/wrench"
