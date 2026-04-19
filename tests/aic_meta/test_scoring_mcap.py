from pathlib import Path

import math

import pytest

from v3_conversion.aic_meta.scoring_mcap import extract_insertion_event


def test_extract_insertion_event_first_message(build_mcap_fixture, tmp_path: Path):
    bag = build_mcap_fixture(
        path=tmp_path / "bag.mcap",
        insertion_event=[(1_000_000_000, "/nic_card_mount_0/sfp_port_0")],
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
