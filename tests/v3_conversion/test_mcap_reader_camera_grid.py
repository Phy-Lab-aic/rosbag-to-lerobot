from v3_conversion.data_spec import Rosbag
from v3_conversion.mcap_reader import extract_frames


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
