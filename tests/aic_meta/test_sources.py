import json
import textwrap
from pathlib import Path

import pytest

from v3_conversion.aic_meta.sources import (
    load_run_meta,
    load_scoring_yaml,
    load_tags,
)


def test_load_run_meta_reads_policy_and_seed(tmp_path: Path):
    (tmp_path / "policy.txt").write_text("cheatcode\n")
    (tmp_path / "seed.txt").write_text("42\n")

    result = load_run_meta(tmp_path)

    assert result == {"policy": "cheatcode", "seed": 42}


def test_load_run_meta_missing_files_yields_empties(tmp_path: Path):
    result = load_run_meta(tmp_path)
    assert result == {"policy": "", "seed": -1}


def test_load_tags_picks_relevant_fields(tmp_path: Path):
    (tmp_path / "tags.json").write_text(
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

    result = load_tags(tmp_path)

    assert result["schema_version"] == "0.1.0"
    assert result["early_term_source"] == "insertion_event"


def test_load_tags_missing_returns_defaults(tmp_path: Path):
    result = load_tags(tmp_path)
    assert result["schema_version"] == ""


def test_load_scoring_yaml_extracts_categories(tmp_path: Path):
    (tmp_path / "scoring.yaml").write_text(
        textwrap.dedent(
            """
            trial_1:
              total: 94.68
              tier_1:
                score: 1.0
                message: Model validation succeeded.
              tier_2:
                score: 18.68
                message: Scoring succeeded.
                categories:
                  contacts: {score: 0.0, message: No contact detected.}
                  duration: {score: 7.63, message: 'Task duration: 25.03 seconds.'}
                  insertion force: {score: 0.0, message: No excessive force detected}
                  trajectory efficiency: {score: 5.88, message: path length info}
                  trajectory smoothness: {score: 5.17, message: jerk info}
              tier_3:
                score: 75.0
                message: Cable insertion successful.
            """
        ).lstrip()
    )

    result = load_scoring_yaml(tmp_path, trial_key="trial_1")

    assert result["score_total"] == pytest.approx(94.68)
    assert result["score_tier1"] == pytest.approx(1.0)
    assert result["score_tier3"] == pytest.approx(75.0)
    assert result["score_contacts"] == pytest.approx(0.0)
    assert result["score_duration_message"] == "Task duration: 25.03 seconds."
    assert result["score_insertion_force_message"] == "No excessive force detected"
    assert result["score_traj_efficiency"] == pytest.approx(5.88)
    assert result["score_traj_smoothness_message"] == "jerk info"
