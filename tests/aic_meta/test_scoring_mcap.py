from pathlib import Path

import math

import pytest

from v3_conversion.aic_meta.scoring_mcap import (
    extract_insertion_event,
    extract_scoring_tf_snapshots,
)


def test_extract_insertion_event_returns_default_for_unreadable_bag(tmp_path: Path):
    bag = tmp_path / "broken.mcap"
    bag.write_bytes(b"not a valid mcap file")

    result = extract_insertion_event(bag, episode_start_ns=0)

    assert result["insertion_event_fired"] is False
    assert result["insertion_event_target"] == ""
    assert math.isnan(result["insertion_event_time_sec"])


def test_extract_insertion_event_first_message(build_mcap_fixture, tmp_path: Path):
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        insertion_event=[(1_000_000_000, "/nic_card_mount_0/sfp_port_0")],
    )

    result = extract_insertion_event(bag, episode_start_ns=500_000_000)

    assert result["insertion_event_fired"] is True
    assert result["insertion_event_target"] == "/nic_card_mount_0/sfp_port_0"
    assert result["insertion_event_time_sec"] == pytest.approx(0.5, abs=1e-6)


def test_extract_insertion_event_uses_first_matching_message(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        insertion_event=[
            (1_000_000_000, "/nic_card_mount_0/sfp_port_0"),
            (1_500_000_000, "/nic_card_mount_0/sfp_port_1"),
        ],
    )

    result = extract_insertion_event(bag, episode_start_ns=500_000_000)

    assert result["insertion_event_fired"] is True
    assert result["insertion_event_target"] == "/nic_card_mount_0/sfp_port_0"
    assert result["insertion_event_time_sec"] == pytest.approx(0.5, abs=1e-6)


def test_extract_insertion_event_absent_is_not_fired(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(path=tmp_path / "bag.mcap")
    result = extract_insertion_event(bag, episode_start_ns=0)
    assert result["insertion_event_fired"] is False
    assert result["insertion_event_target"] == ""
    assert math.isnan(result["insertion_event_time_sec"])


def test_extract_insertion_event_skips_missing_decoder(
    build_mcap_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        insertion_event=[(1_000_000_000, "/nic_card_mount_0/sfp_port_0")],
    )

    class FakeFactory:
        def decoder_for(self, message_encoding, schema):
            return None

    from v3_conversion.aic_meta import scoring_mcap

    monkeypatch.setattr(scoring_mcap, "DecoderFactory", lambda: FakeFactory())

    result = extract_insertion_event(bag, episode_start_ns=0)

    assert result["insertion_event_fired"] is False
    assert result["insertion_event_target"] == ""
    assert math.isnan(result["insertion_event_time_sec"])


def test_extract_insertion_event_skips_decode_failure(
    build_mcap_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        insertion_event=[(1_000_000_000, "/nic_card_mount_0/sfp_port_0")],
    )

    class FakeFactory:
        def decoder_for(self, message_encoding, schema):
            def _decode(_data):
                raise ValueError("boom")

            return _decode

    from v3_conversion.aic_meta import scoring_mcap

    monkeypatch.setattr(scoring_mcap, "DecoderFactory", lambda: FakeFactory())

    result = extract_insertion_event(bag, episode_start_ns=0)

    assert result["insertion_event_fired"] is False
    assert result["insertion_event_target"] == ""
    assert math.isnan(result["insertion_event_time_sec"])


def test_scoring_tf_snapshots_capture_initial_and_final(
    build_mcap_fixture, tmp_path: Path
):
    initial_tf = (
        0,
        [
            ("world", "task_board", 0.15, -0.20, 1.14, 0.0, 0.0, 0.0, 1.0),
            ("task_board", "nic_card_mount_0", 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )
    mid_tf = (
        500_000_000,
        [
            ("task_board", "nic_card_mount_0", 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )
    final_tf = (
        10_000_000_000,
        [
            ("world", "task_board", 0.15, -0.20, 1.14, 0.0, 0.0, 0.0, 1.0),
            ("task_board", "nic_card_mount_0", 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        scoring_tf=[initial_tf, mid_tf, final_tf],
    )

    result = extract_scoring_tf_snapshots(bag, window_ns=1_000_000_000)

    initial_frames = {f["frame_id"]: f for f in result["scoring_frames_initial"]}
    assert "task_board" in initial_frames
    assert initial_frames["nic_card_mount_0"]["pose"][0] == pytest.approx(0.02)

    final_frames = {f["frame_id"]: f for f in result["scoring_frames_final"]}
    assert final_frames["task_board"]["parent_frame_id"] == "world"
    assert final_frames["nic_card_mount_0"]["pose"][0] == pytest.approx(0.03)


def test_scoring_tf_snapshots_return_empty_for_unreadable_bag(tmp_path: Path):
    bag = tmp_path / "broken.mcap"
    bag.write_bytes(b"not a valid mcap file")

    result = extract_scoring_tf_snapshots(bag)

    assert result == {"scoring_frames_initial": [], "scoring_frames_final": []}


def test_scoring_tf_snapshots_skip_missing_decoder(
    build_mcap_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        scoring_tf=[
            (
                0,
                [
                    (
                        "world",
                        "task_board",
                        0.15,
                        -0.20,
                        1.14,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    )
                ],
            )
        ],
    )

    class FakeFactory:
        def decoder_for(self, message_encoding, schema):
            return None

    from v3_conversion.aic_meta import scoring_mcap

    monkeypatch.setattr(scoring_mcap, "DecoderFactory", lambda: FakeFactory())

    result = extract_scoring_tf_snapshots(bag)

    assert result == {"scoring_frames_initial": [], "scoring_frames_final": []}


def test_scoring_tf_snapshots_skip_decode_failure(
    build_mcap_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        scoring_tf=[
            (
                0,
                [
                    (
                        "world",
                        "task_board",
                        0.15,
                        -0.20,
                        1.14,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    )
                ],
            )
        ],
    )

    class FakeFactory:
        def decoder_for(self, message_encoding, schema):
            def _decode(_data):
                raise ValueError("boom")

            return _decode

    from v3_conversion.aic_meta import scoring_mcap

    monkeypatch.setattr(scoring_mcap, "DecoderFactory", lambda: FakeFactory())

    result = extract_scoring_tf_snapshots(bag)

    assert result == {"scoring_frames_initial": [], "scoring_frames_final": []}
