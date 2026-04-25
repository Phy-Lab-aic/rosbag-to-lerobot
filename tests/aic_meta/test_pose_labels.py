import math
from pathlib import Path

import numpy as np

from v3_conversion.aic_meta.pose_labels import (
    extract_pose_labels,
    frame_id_candidates,
)


def test_frame_id_candidates_cover_real_aic_suffixes():
    candidates = frame_id_candidates(
        name="sfp_tip",
        cable_name="cable_0",
        target_module="nic_card_mount_0",
    )

    assert "sfp_tip" in candidates
    assert "sfp_tip_link" in candidates
    assert "cable_0/sfp_tip_link" in candidates


def test_extract_pose_labels_uses_controller_state_for_tcp(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "tcp_controller.mcap",
        controller_state=[
            (0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            (50_000_000, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )

    labels = extract_pose_labels(
        bag_path=bag,
        frame_timestamps_ns=[0, 50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert np.allclose(
        labels["label.tcp_pose"][0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    )
    assert labels["label.tcp_pose_valid"].tolist() == [True, True]


def test_extract_pose_labels_uses_tf_fallback_for_tcp(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "tcp_tf.mcap",
        tf=[
            (
                0,
                [("base_link", "tcp_link", 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)],
            ),
        ],
    )

    labels = extract_pose_labels(
        bag_path=bag,
        frame_timestamps_ns=[0],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert np.allclose(
        labels["label.tcp_pose"][0], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    )
    assert labels["label.tcp_pose_valid"].tolist() == [True]


def test_extract_pose_labels_prioritizes_controller_state_over_tf_for_tcp(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "tcp_priority.mcap",
        controller_state=[
            (50_000_000, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0),
        ],
        tf=[
            (
                0,
                [("base_link", "tcp_link", 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)],
            ),
            (
                50_000_000,
                [("base_link", "tcp_link", 9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0)],
            ),
        ],
    )

    labels = extract_pose_labels(
        bag_path=bag,
        frame_timestamps_ns=[0, 50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert labels["label.tcp_pose_valid"].tolist() == [False, True]
    assert math.isnan(float(labels["label.tcp_pose"][0][0]))
    assert np.allclose(
        labels["label.tcp_pose"][1], [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
    )


def test_extract_pose_labels_samples_scoring_tf_and_masks_missing(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "scoring_labels.mcap",
        scoring_tf=[
            (
                50_000_000,
                [
                    (
                        "base_link",
                        "cable_0/sfp_tip_link",
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    (
                        "base_link",
                        "task_board/nic_card_mount_0/sfp_port_0_link",
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    (
                        "base_link",
                        "task_board/nic_card_mount_0",
                        3.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                ],
            )
        ],
    )

    labels = extract_pose_labels(
        bag_path=bag,
        frame_timestamps_ns=[0, 50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert labels["label.plug_pose_base_valid"].tolist() == [False, True]
    assert math.isnan(float(labels["label.plug_pose_base"][0][0]))
    assert np.allclose(
        labels["label.plug_pose_base"][1], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    assert np.allclose(
        labels["label.port_pose_base"][1], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    assert np.allclose(
        labels["label.target_module_pose_base"][1],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )
