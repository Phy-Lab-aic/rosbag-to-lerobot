from pathlib import Path

from v3_conversion.aic_meta.sources import load_run_meta


def test_load_run_meta_reads_policy_and_seed(tmp_path: Path):
    (tmp_path / "policy.txt").write_text("cheatcode\n")
    (tmp_path / "seed.txt").write_text("42\n")

    result = load_run_meta(tmp_path)

    assert result == {"policy": "cheatcode", "seed": 42}


def test_load_run_meta_missing_files_yields_empties(tmp_path: Path):
    result = load_run_meta(tmp_path)
    assert result == {"policy": "", "seed": -1}
