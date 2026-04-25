from pathlib import Path

import pyarrow.parquet as pq
import pytest

from v3_conversion.aic_meta.pose_commands import extract_pose_commands
from v3_conversion.aic_meta.writer import write_pose_commands_parquet


def test_extract_pose_commands_preserves_sparse_times(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "commands.mcap",
        pose_commands=[
            (
                1_000_000_000,
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                [90.0] * 36,
                [20.0] * 36,
            ),
            (
                1_500_000_000,
                4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                [91.0] * 36,
                [21.0] * 36,
            ),
        ],
    )

    rows = extract_pose_commands(
        bag_path=bag,
        episode_index=7,
        episode_start_ns=1_000_000_000,
    )

    assert [row["episode_index"] for row in rows] == [7, 7]
    assert [row["t_ns"] for row in rows] == [1_000_000_000, 1_500_000_000]
    assert rows[0]["time_sec"] == pytest.approx(0.0)
    assert rows[1]["time_sec"] == pytest.approx(0.5)
    assert rows[0]["pose"] == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    assert rows[0]["velocity"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert rows[0]["stiffness"] == [90.0] * 36
    assert rows[0]["damping"] == [20.0] * 36


def test_write_pose_commands_parquet_roundtrip(tmp_path: Path):
    target = tmp_path / "aic" / "pose_commands.parquet"
    rows = [
        {
            "episode_index": 7,
            "t_ns": 1_000_000_000,
            "time_sec": 0.0,
            "pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            "velocity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "stiffness": [90.0] * 36,
            "damping": [20.0] * 36,
        }
    ]

    write_pose_commands_parquet(target, rows)

    loaded = pq.read_table(target).to_pylist()
    assert loaded[0]["episode_index"] == 7
    assert loaded[0]["pose"][0] == pytest.approx(1.0)
    assert len(loaded[0]["stiffness"]) == 36
