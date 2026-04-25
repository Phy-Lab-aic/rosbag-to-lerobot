from v3_conversion.data_spec import Rosbag
from v3_conversion.mcap_reader import build_extraction_config, extract_frames


JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper/left_finger_joint",
]


def _mk_bag(build_mcap_fixture, tmp_path):
    h, w = 2, 2
    img_bytes = bytes([0] * (h * w * 3))
    cam_times = [0, 50_000_000, 100_000_000]
    images = {
        "/left_camera/image": [(t, h, w, img_bytes) for t in cam_times],
        "/center_camera/image": [(t, h, w, img_bytes) for t in cam_times],
        "/right_camera/image": [(t, h, w, img_bytes) for t in cam_times],
    }
    joint_states = []
    for i in range(0, 300):
        t_ns = i * 2_000_000
        pos = [i * 0.001] * 7
        joint_states.append((t_ns, JOINTS, pos))
    wrench = [(i * 20_000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) for i in range(20)]

    return build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        joint_states=joint_states,
        wrench=wrench,
        images=images,
    )


def _pixel_frame(value):
    return bytes([value, value, value])


def _mk_sync_images(cam_times_by_topic):
    h, w = 1, 1
    images = {}
    for topic, times in cam_times_by_topic.items():
        frames = []
        for idx, t_ns in enumerate(times):
            frames.append((t_ns, h, w, _pixel_frame(10 + idx)))
        images[topic] = frames
    return images


def test_extract_frames_camera_grid_three_frames(build_mcap_fixture, tmp_path):
    bag = _mk_bag(build_mcap_fixture, tmp_path)
    config = Rosbag(
        topic_map={
            "/left_camera/image": "cam_left",
            "/center_camera/image": "cam_center",
            "/right_camera/image": "cam_right",
            "/joint_states": "observation",
            "/fts_broadcaster/wrench": "wrench",
        },
        action_order=["action"],
        joint_order={"obs": JOINTS, "action": {"action": JOINTS}},
        camera_names=["cam_left", "cam_center", "cam_right"],
        fps=20,
        shared_action_names=["action"],
        wrench_topic="/fts_broadcaster/wrench",
    )
    frames, _ = extract_frames(bag_path=str(bag), config=config)
    assert len(frames) == 3
    for frame in frames:
        assert frame["obs"].shape == (7,)
        assert frame["wrench"].shape == (6,)
        assert set(frame["images"]) == {"cam_left", "cam_center", "cam_right"}


def test_extract_frames_requires_all_cameras_on_same_tick(build_mcap_fixture, tmp_path):
    bag = build_mcap_fixture(
        path=tmp_path / "lagging_camera.mcap",
        joint_states=[(i * 2_000_000, JOINTS, [i * 0.001] * 7) for i in range(80)],
        wrench=[(i * 20_000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) for i in range(10)],
        images=_mk_sync_images(
            {
                "/left_camera/image": [0, 50_000_000, 100_000_000],
                "/center_camera/image": [0, 100_000_000],
                "/right_camera/image": [0, 100_000_000],
            }
        ),
    )
    config = Rosbag(
        topic_map={
            "/left_camera/image": "cam_left",
            "/center_camera/image": "cam_center",
            "/right_camera/image": "cam_right",
            "/joint_states": "observation",
            "/fts_broadcaster/wrench": "wrench",
        },
        action_order=["action"],
        joint_order={"obs": JOINTS, "action": {"action": JOINTS}},
        camera_names=["cam_left", "cam_center", "cam_right"],
        fps=20,
        shared_action_names=["action"],
        wrench_topic="/fts_broadcaster/wrench",
    )

    frames, _ = extract_frames(bag_path=str(bag), config=config)

    assert len(frames) == 2
    assert [frame["images"]["cam_left"][0, 0, 0] for frame in frames] == [10, 12]
    assert [frame["images"]["cam_center"][0, 0, 0] for frame in frames] == [10, 11]
    assert [frame["images"]["cam_right"][0, 0, 0] for frame in frames] == [10, 11]


def test_extract_frames_accepts_camera_jitter_within_half_frame(
    build_mcap_fixture, tmp_path
):
    bag = build_mcap_fixture(
        path=tmp_path / "camera_jitter.mcap",
        joint_states=[(i * 1_000_000, JOINTS, [i * 0.001] * 7) for i in range(140)],
        wrench=[(i * 10_000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) for i in range(20)],
        images=_mk_sync_images(
            {
                "/left_camera/image": [10_000_000, 60_000_000, 110_000_000],
                "/center_camera/image": [7_000_000, 57_000_000, 107_000_000],
                "/right_camera/image": [13_000_000, 63_000_000, 113_000_000],
            }
        ),
    )
    config = Rosbag(
        topic_map={
            "/left_camera/image": "cam_left",
            "/center_camera/image": "cam_center",
            "/right_camera/image": "cam_right",
            "/joint_states": "observation",
            "/fts_broadcaster/wrench": "wrench",
        },
        action_order=["action"],
        joint_order={"obs": JOINTS, "action": {"action": JOINTS}},
        camera_names=["cam_left", "cam_center", "cam_right"],
        fps=20,
        shared_action_names=["action"],
        wrench_topic="/fts_broadcaster/wrench",
    )

    frames, _ = extract_frames(bag_path=str(bag), config=config)

    assert len(frames) == 3
    assert [frame["emitted_timestamp_ns"] for frame in frames] == [
        10_000_000,
        60_000_000,
        110_000_000,
    ]
    assert [frame["images"]["cam_left"][0, 0, 0] for frame in frames] == [10, 11, 12]
    assert [frame["images"]["cam_center"][0, 0, 0] for frame in frames] == [10, 11, 12]
    assert [frame["images"]["cam_right"][0, 0, 0] for frame in frames] == [10, 11, 12]


def test_extract_frames_uses_dedicated_action_topics(build_mcap_fixture, tmp_path):
    cam_times = [0, 50_000_000, 100_000_000]
    bag = build_mcap_fixture(
        path=tmp_path / "dedicated_action.mcap",
        joint_states=[(i * 2_000_000, JOINTS, [i * 0.001] * 7) for i in range(80)],
        joint_state_topics={
            "/leader_joint_states": [
                (t_ns, JOINTS, [0.5 + idx] * 7) for idx, t_ns in enumerate(cam_times)
            ]
        },
        images=_mk_sync_images(
            {
                "/left_camera/image": cam_times,
                "/center_camera/image": cam_times,
                "/right_camera/image": cam_times,
            }
        ),
    )
    config = Rosbag(
        topic_map={
            "/left_camera/image": "cam_left",
            "/center_camera/image": "cam_center",
            "/right_camera/image": "cam_right",
            "/joint_states": "observation",
            "/leader_joint_states": "action",
        },
        action_order=["action"],
        joint_order={"obs": JOINTS, "action": {"action": JOINTS}},
        camera_names=["cam_left", "cam_center", "cam_right"],
        fps=20,
        shared_action_names=[],
    )

    frames, _ = extract_frames(bag_path=str(bag), config=config)

    assert len(frames) == 3
    assert [frame["action"]["action"][0] for frame in frames] == [0.5, 1.5, 2.5]


def test_extract_frames_downsamples_primary_camera_to_config_fps(build_mcap_fixture, tmp_path):
    cam_times = [0, 25_000_000, 50_000_000, 75_000_000, 100_000_000]
    bag = build_mcap_fixture(
        path=tmp_path / "fps_gated.mcap",
        joint_states=[(i * 1_000_000, JOINTS, [i * 0.001] * 7) for i in range(120)],
        wrench=[(i * 10_000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) for i in range(20)],
        images=_mk_sync_images(
            {
                "/left_camera/image": cam_times,
                "/center_camera/image": cam_times,
                "/right_camera/image": cam_times,
            }
        ),
    )
    config = Rosbag(
        topic_map={
            "/left_camera/image": "cam_left",
            "/center_camera/image": "cam_center",
            "/right_camera/image": "cam_right",
            "/joint_states": "observation",
            "/fts_broadcaster/wrench": "wrench",
        },
        action_order=["action"],
        joint_order={"obs": JOINTS, "action": {"action": JOINTS}},
        camera_names=["cam_left", "cam_center", "cam_right"],
        fps=20,
        shared_action_names=["action"],
        wrench_topic="/fts_broadcaster/wrench",
    )

    frames, _ = extract_frames(bag_path=str(bag), config=config)

    assert len(frames) == 3
    assert [round(float(frame["obs"][0]), 3) for frame in frames] == [0.0, 0.05, 0.1]


def test_extract_frames_emits_when_non_primary_camera_closes_tick(
    build_mcap_fixture, tmp_path
):
    cam_times = [0, 50_000_000, 100_000_000]
    bag = build_mcap_fixture(
        path=tmp_path / "non_primary_closes_tick.mcap",
        joint_states=[(i * 2_000_000, JOINTS, [i * 0.001] * 7) for i in range(80)],
        images={
            "/cam_b/image": [(t, 1, 1, _pixel_frame(20 + idx)) for idx, t in enumerate(cam_times)],
            "/cam_c/image": [(t, 1, 1, _pixel_frame(30 + idx)) for idx, t in enumerate(cam_times)],
            "/cam_a/image": [(t, 1, 1, _pixel_frame(10 + idx)) for idx, t in enumerate(cam_times)],
        },
    )
    config = build_extraction_config(
        detail={
            "camera_topic_map": {
                "cam_a": "/cam_a/image",
                "cam_b": "/cam_b/image",
                "cam_c": "/cam_c/image",
            },
            "joint_names": JOINTS,
            "action_topics_map": {"leader": "/joint_states"},
            "state_topic": "/joint_states",
        },
        fps=20,
        robot_type="",
    )

    frames, _ = extract_frames(bag_path=str(bag), config=config)

    assert len(frames) == 3
    assert [frame["images"]["cam_a"][0, 0, 0] for frame in frames] == [10, 11, 12]
