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


def _pixel_frame(value):
    return bytes([value, value, value])


def test_extract_frames_tags_emitted_sync_timestamp(build_mcap_fixture, tmp_path):
    cam_times = [0, 50_000_000, 100_000_000]
    bag = build_mcap_fixture(
        path=tmp_path / "camera_preroll.mcap",
        joint_states=[(i * 2_000_000, JOINTS, [i * 0.001] * 7) for i in range(80)],
        images={
            "/left_camera/image": [(t, 1, 1, _pixel_frame(10 + idx)) for idx, t in enumerate(cam_times)],
            "/center_camera/image": [(t, 1, 1, _pixel_frame(20 + idx)) for idx, t in enumerate(cam_times[1:])],
            "/right_camera/image": [(t, 1, 1, _pixel_frame(30 + idx)) for idx, t in enumerate(cam_times[1:])],
        },
    )
    config = Rosbag(
        topic_map={
            "/left_camera/image": "cam_left",
            "/center_camera/image": "cam_center",
            "/right_camera/image": "cam_right",
            "/joint_states": "observation",
        },
        action_order=["action"],
        joint_order={"obs": JOINTS, "action": {"action": JOINTS}},
        camera_names=["cam_left", "cam_center", "cam_right"],
        fps=20,
        shared_action_names=["action"],
    )

    frames, timestamps = extract_frames(bag_path=str(bag), config=config)

    assert timestamps["cam_left"] == [0, 50_000_000, 100_000_000]
    assert [frame["emitted_timestamp_ns"] for frame in frames] == [50_000_000, 100_000_000]
