import json
from pathlib import Path

from v3_conversion.aic_meta.sources import load_validation_status


def test_load_validation_status_accepts_all_passed(tmp_path: Path):
    (tmp_path / "validation.json").write_text(
        json.dumps({"passed_count": 3, "total_count": 3, "checks": []})
    )

    result = load_validation_status(tmp_path)

    assert result["passed"] is True
    assert result["reason"] == ""


def test_load_validation_status_rejects_missing_file(tmp_path: Path):
    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json missing"


def test_load_validation_status_rejects_failed_counts(tmp_path: Path):
    (tmp_path / "validation.json").write_text(
        json.dumps({"passed_count": 2, "total_count": 3, "checks": []})
    )

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json passed_count 2 != total_count 3"


def test_load_validation_status_rejects_failed_check(tmp_path: Path):
    (tmp_path / "validation.json").write_text(
        json.dumps(
            {
                "passed_count": 2,
                "total_count": 2,
                "checks": [
                    {"name": "config.yaml", "passed": True},
                    {"name": "episode/metadata.json", "passed": False},
                ],
            }
        )
    )

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json failed check: episode/metadata.json"


def test_load_validation_status_rejects_non_object_json(tmp_path: Path):
    (tmp_path / "validation.json").write_text(json.dumps([]))

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json invalid: expected object"


def test_load_validation_status_rejects_missing_required_keys(tmp_path: Path):
    (tmp_path / "validation.json").write_text(json.dumps({"total_count": 3, "checks": []}))

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json invalid: missing passed_count"


def test_load_validation_status_rejects_first_missing_key_in_order(tmp_path: Path):
    (tmp_path / "validation.json").write_text(json.dumps({"passed_count": 3}))

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json invalid: missing total_count"


def test_load_validation_status_rejects_non_list_checks(tmp_path: Path):
    (tmp_path / "validation.json").write_text(
        json.dumps({"passed_count": 3, "total_count": 3, "checks": {}})
    )

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json invalid: checks must be list"


def test_load_validation_status_rejects_invalid_json(tmp_path: Path):
    (tmp_path / "validation.json").write_text("{")

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json invalid json"
