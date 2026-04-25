import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest


JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper/left_finger_joint",
]


class _FakeDataset:
    def __init__(self):
        self.meta = SimpleNamespace(total_episodes=0)

    def finalize(self):
        return None


class _FakeDataCreator:
    instances = []

    def __init__(self, **kwargs):
        self.dataset = None
        self.episodes = []
        _FakeDataCreator.instances.append(self)

    def convert_episode(self, episode):
        self.episodes.append(episode)
        if self.dataset is None:
            self.dataset = _FakeDataset()
        self.dataset.meta.total_episodes += 1

    def recover_dataset_state(self):
        return None

    def correct_video_timestamps(self):
        return None

    def patch_episodes_metadata(self):
        return None


def _import_main(monkeypatch):
    sys.modules.pop("main", None)
    data_creator_mod = types.ModuleType("v3_conversion.data_creator")
    data_creator_mod.DataCreator = _FakeDataCreator
    monkeypatch.setitem(sys.modules, "v3_conversion.data_creator", data_creator_mod)
    module = importlib.import_module("main")
    return importlib.reload(module)


def test_run_conversion_produces_split_aic_parquets(
    build_mcap_fixture, tmp_path: Path, monkeypatch
):
    _FakeDataCreator.instances = []
    run_dir = tmp_path / "input" / "run_test_20260420_000000"
    trial_dir = run_dir / "trial_1_score95"
    episode_dir = trial_dir / "episode"
    episode_dir.mkdir(parents=True)

    (run_dir / "validation.json").write_text(
        json.dumps(
            {
                "passed_count": 1,
                "total_count": 1,
                "checks": [{"name": "episode/metadata.json", "passed": True}],
            }
        )
    )
    (run_dir / "policy.txt").write_text("cheatcode\n")
    (run_dir / "seed.txt").write_text("42\n")
    (run_dir / "config.yaml").write_text(
        "trials:\n"
        "  trial_1:\n"
        "    scene:\n"
        "      task_board:\n"
        "        nic_rail_0: {entity_present: true, entity_name: nic_card_0}\n"
        "      cables:\n"
        "        cable_0:\n"
        "          pose:\n"
        "            gripper_offset: {x: 0.0, y: 0.01, z: 0.04}\n"
        "            roll: 0.4\n"
        "            pitch: -0.4\n"
        "            yaw: 1.3\n"
    )
    (trial_dir / "tags.json").write_text(
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
    (trial_dir / "scoring.yaml").write_text(
        "trial_1:\n"
        "  total: 94.68\n"
        "  tier_1: {score: 1.0, message: ok}\n"
        "  tier_2:\n"
        "    score: 18.68\n"
        "    message: ok\n"
        "    categories:\n"
        "      contacts: {score: 0.0, message: ok}\n"
        "      duration: {score: 7.63, message: d}\n"
        "      insertion force: {score: 0.0, message: f}\n"
        "      trajectory efficiency: {score: 5.88, message: e}\n"
        "      trajectory smoothness: {score: 5.17, message: s}\n"
        "  tier_3: {score: 75.0, message: Cable insertion successful.}\n"
    )
    (episode_dir / "metadata.json").write_text(
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
                "num_steps": 3,
                "duration_sec": 0.1,
            }
        )
    )

    h, w = 2, 2
    img_bytes = bytes([0] * (h * w * 3))
    cam_times = [0, 50_000_000, 100_000_000]
    images = {
        "/left_camera/image": [(t, h, w, img_bytes) for t in cam_times],
        "/center_camera/image": [(t, h, w, img_bytes) for t in cam_times],
        "/right_camera/image": [(t, h, w, img_bytes) for t in cam_times],
    }
    joint_states = [
        (i * 2_000_000, JOINTS, [i * 0.001] * 7, [i * 0.01] * 7)
        for i in range(150)
    ]
    wrench = [
        (i * 20_000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) for i in range(7)
    ]
    insertion_event = [(90_000_000, "/nic_card_mount_0/sfp_port_0")]
    scoring_tf = [
        (0, [("world", "task_board", 0.15, -0.2, 1.14, 0.0, 0.0, 0.0, 1.0)]),
        (
            100_000_000,
            [("task_board", "nic_card_mount_0", 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        ),
    ]
    controller_state = [
        (0, 0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0),
        (50_000_000, 0.6, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0),
    ]
    pose_commands = [
        (
            60_000_000,
            0.7, 0.1, 0.2,
            0.0, 0.0, 0.0, 1.0,
            0.01, 0.02, 0.03,
            0.0, 0.0, 0.1,
            [100.0, 100.0],
            [10.0, 10.0],
        )
    ]
    build_mcap_fixture(
        path=run_dir / f"{run_dir.name}_0.mcap",
        joint_states=joint_states,
        wrench=wrench,
        images=images,
        insertion_event=insertion_event,
        scoring_tf=scoring_tf,
        controller_state=controller_state,
        pose_commands=pose_commands,
    )

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "task": "aic_test",
                "robot": "ur5e",
                "fps": 20,
                "folders": [run_dir.name],
                "camera_topic_map": {
                    "cam_left": "/left_camera/image",
                    "cam_center": "/center_camera/image",
                    "cam_right": "/right_camera/image",
                },
                "joint_names": JOINTS,
                "state_topic": "/joint_states",
                "wrench_topic": "/fts_broadcaster/wrench",
                "action_topics_map": {"leader": "/joint_states"},
            }
        )
    )

    main = _import_main(monkeypatch)
    out_dir = tmp_path / "out"
    rc = main.run_conversion(
        config_path=str(cfg_path),
        input_dir=str(run_dir.parent),
        output_dir=str(out_dir),
    )

    assert rc == 0

    aic_dir = out_dir / "aic_test" / "meta" / "aic"
    assert (aic_dir / "task.parquet").is_file()
    assert (aic_dir / "scoring.parquet").is_file()
    assert (aic_dir / "scene.parquet").is_file()
    assert (aic_dir / "tf_snapshots.parquet").is_file()
    assert (aic_dir / "pose_commands.parquet").is_file()

    converted_episode = _FakeDataCreator.instances[0].episodes[0]
    assert "velocity" in converted_episode
    assert "label.tcp_pose" in converted_episode
    assert "label.tcp_pose_valid" in converted_episode
    assert converted_episode["label.tcp_pose_valid"].tolist() == [True, True]

    task_row = pq.read_table(aic_dir / "task.parquet").to_pylist()[0]
    assert task_row["episode_index"] == 0
    assert task_row["cable_type"] == "sfp_sc"
    assert task_row["policy"] == "cheatcode"
    assert task_row["seed"] == 42
    assert task_row["insertion_event_fired"] is True
    assert task_row["insertion_event_target"] == "/nic_card_mount_0/sfp_port_0"

    scoring_row = pq.read_table(aic_dir / "scoring.parquet").to_pylist()[0]
    assert scoring_row["score_total"] == pytest.approx(94.68, abs=1e-6)
    assert scoring_row["score_traj_efficiency_message"] == "e"

    scene_row = pq.read_table(aic_dir / "scene.parquet").to_pylist()[0]
    assert scene_row["plug_port_distance_init"] == pytest.approx(0.001, abs=1e-6)
    assert scene_row["scene_rails"][0]["name"] == "nic_rail_0"

    tf_row = pq.read_table(aic_dir / "tf_snapshots.parquet").to_pylist()[0]
    initial_frames = {frame["frame_id"]: frame for frame in tf_row["scoring_frames_initial"]}
    final_frames = {frame["frame_id"]: frame for frame in tf_row["scoring_frames_final"]}
    assert initial_frames["task_board"]["parent_frame_id"] == "world"
    assert final_frames["nic_card_mount_0"]["parent_frame_id"] == "task_board"

    pose_command_row = pq.read_table(aic_dir / "pose_commands.parquet").to_pylist()[0]
    assert pose_command_row["episode_index"] == 0
    assert pose_command_row["t_ns"] == 60_000_000
