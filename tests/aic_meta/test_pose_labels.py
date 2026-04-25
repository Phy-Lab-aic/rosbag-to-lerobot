import math
from pathlib import Path

import numpy as np

from v3_conversion.aic_meta import pose_labels
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


def test_frame_id_candidates_include_stripped_target_module_aliases():
    candidates = frame_id_candidates(
        name="nic_card_mount_0",
        target_module="nic_card_mount_0",
    )

    assert "task_board/nic_card_mount_0/nic_card_mount_link" in candidates


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


def test_extract_pose_labels_composes_chained_tf_fallback_for_tcp(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "tcp_tf_chain.mcap",
        tf=[
            (
                0,
                [
                    (
                        "base_link",
                        "wrist_3_link",
                        0.1,
                        0.2,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    (
                        "wrist_3_link",
                        "tcp_link",
                        0.4,
                        0.5,
                        0.6,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                ],
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

    assert labels["label.tcp_pose_valid"].tolist() == [True]
    assert np.allclose(
        labels["label.tcp_pose"][0],
        [0.5, 0.7, 0.9, 0.0, 0.0, 0.0, 1.0],
    )


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


def test_extract_pose_labels_composes_scoring_tf_chain_to_base(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "scoring_chain.mcap",
        scoring_tf=[
            (
                50_000_000,
                [
                    (
                        "base_link",
                        "task_board",
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    (
                        "task_board",
                        "task_board/nic_card_mount_0",
                        0.0,
                        2.0,
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
        frame_timestamps_ns=[50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert labels["label.target_module_pose_base_valid"].tolist() == [True]
    assert np.allclose(
        labels["label.target_module_pose_base"][0],
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )


def test_extract_pose_labels_leaves_orphan_non_base_scoring_tf_invalid(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "scoring_orphan.mcap",
        scoring_tf=[
            (
                50_000_000,
                [
                    (
                        "task_board",
                        "task_board/nic_card_mount_0",
                        0.0,
                        2.0,
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
        frame_timestamps_ns=[50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert labels["label.target_module_pose_base_valid"].tolist() == [False]
    assert math.isnan(float(labels["label.target_module_pose_base"][0][0]))


def test_extract_pose_labels_uses_ordered_candidate_preference(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "scoring_preference.mcap",
        scoring_tf=[
            (
                50_000_000,
                [
                    (
                        "base_link",
                        "task_board/nic_card_mount_0",
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
                        "task_board/nic_card_mount_0/nic_card_mount_0_link",
                        2.0,
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
        frame_timestamps_ns=[50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert labels["label.target_module_pose_base_valid"].tolist() == [True]
    assert np.allclose(
        labels["label.target_module_pose_base"][0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )


def test_decoded_messages_warns_when_decoder_construction_fails(
    tmp_path: Path, monkeypatch, caplog
):
    class Schema:
        def __init__(self, record_id):
            self.id = record_id

    class Channel:
        def __init__(self, record_id, topic, schema_id):
            self.id = record_id
            self.topic = topic
            self.schema_id = schema_id
            self.message_encoding = "cdr"

    class Message:
        def __init__(self, channel_id, log_time, data):
            self.channel_id = channel_id
            self.log_time = log_time
            self.data = data

    class FakeStreamReader:
        def __init__(self, _stream, record_size_limit=None):
            self.records = [
                Schema(1),
                Schema(2),
                Channel(1, "/tf", 1),
                Channel(2, "/scoring/tf", 2),
                Message(1, 10, b"bad"),
                Message(2, 20, b"good"),
            ]

    class FakeFactory:
        def decoder_for(self, _message_encoding, schema):
            if schema.id == 1:
                raise ValueError("bad decoder")

            return lambda data: data.decode()

    bag = tmp_path / "fake.mcap"
    bag.write_bytes(b"fake")
    monkeypatch.setattr(pose_labels, "StreamReader", FakeStreamReader)
    monkeypatch.setattr(pose_labels, "DecoderFactory", lambda: FakeFactory())
    monkeypatch.setattr(pose_labels.logger, "propagate", True)

    caplog.set_level("WARNING", logger="v3_conversion.aic_meta.pose_labels")

    messages = list(pose_labels._decoded_messages(bag, {"/tf", "/scoring/tf"}))

    assert messages == [("/scoring/tf", 20, "good")]
    assert "Unable to construct decoder" in caplog.text
    assert "/tf" in caplog.text
    assert "bad decoder" in caplog.text
