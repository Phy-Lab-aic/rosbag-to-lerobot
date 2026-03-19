"""Hz validation for the standalone conversion pipeline.

Pure Python — no ROS2 or MCAP dependencies.
validate_mcap_topics() has been moved to mcap_reader.py.
"""

from typing import Any

NSEC_PER_SEC = 1_000_000_000.0


class HzValidationResult:
    """Validation result for entire MCAP file."""

    def __init__(
        self,
        is_valid: bool,
        overall_message: str,
        topic_results: dict[str, dict[str, Any]] | None = None,
        timing_source: str = "",
        timing_source_valid: bool = True,
    ):
        self.is_valid = is_valid
        self.overall_message = overall_message
        self.topic_results = topic_results or {}
        self.timing_source = timing_source
        self.timing_source_valid = timing_source_valid

    def format_diagnostic(self) -> str:
        """Format a human-readable diagnostic message."""
        lines = [self.overall_message, ""]

        if self.timing_source:
            status = "PASS" if self.timing_source_valid else "FAIL"
            lines.append(f"Timing source: {self.timing_source} [{status}]")
            lines.append("")

        failed = [r for r in self.topic_results.values() if not r["is_valid"]]
        passed = [r for r in self.topic_results.values() if r["is_valid"]]

        if failed:
            lines.append("FAILED topics:")
            for r in failed:
                lines.append(
                    f"  - {r['topic']}: {r['actual_hz']:.1f} Hz "
                    f"(min: {r['min_hz']:.1f} Hz, target: {r['target_hz']:.1f} Hz)"
                )
                lines.append(f"    {r['message']}")
            lines.append("")

        if passed:
            lines.append("PASSED topics:")
            for r in passed:
                lines.append(f"  - {r['topic']}: {r['actual_hz']:.1f} Hz")

        return "\n".join(lines)


def _validate_topic_hz(
    timestamps: list[int],
    target_hz: float,
    min_ratio: float,
    topic: str,
) -> dict[str, Any]:
    """Validate timestamps for a single topic and compute actual frequency."""
    min_hz = target_hz * min_ratio
    message_count = len(timestamps)

    if message_count < 2:
        return {
            "topic": topic,
            "is_valid": False,
            "actual_hz": 0.0,
            "min_hz": min_hz,
            "target_hz": target_hz,
            "message_count": message_count,
            "duration_sec": 0.0,
            "message": f"Insufficient messages: {message_count} (need at least 2)",
        }

    min_ts = min(timestamps)
    max_ts = max(timestamps)
    duration_ns = max_ts - min_ts
    duration_sec = duration_ns / NSEC_PER_SEC

    if duration_sec <= 0:
        return {
            "topic": topic,
            "is_valid": False,
            "actual_hz": 0.0,
            "min_hz": min_hz,
            "target_hz": target_hz,
            "message_count": message_count,
            "duration_sec": 0.0,
            "message": "Zero duration (all messages have same timestamp)",
        }

    actual_hz = (message_count - 1) / duration_sec
    is_valid = actual_hz >= min_hz
    message = (
        f"PASS: {actual_hz:.1f} Hz >= {min_hz:.1f} Hz"
        if is_valid
        else f"FAIL: {actual_hz:.1f} Hz < {min_hz:.1f} Hz"
    )

    return {
        "topic": topic,
        "is_valid": is_valid,
        "actual_hz": actual_hz,
        "min_hz": min_hz,
        "target_hz": target_hz,
        "message_count": message_count,
        "duration_sec": duration_sec,
        "message": message,
    }


def _build_overall_result(
    topic_results: dict[str, dict[str, Any]],
    timing_source: str,
    target_hz: float,
    min_ratio: float,
    validate_all_topics: bool,
) -> tuple[bool, str, bool]:
    timing_source_result = topic_results[timing_source]
    timing_source_valid = timing_source_result["is_valid"]

    if validate_all_topics:
        failed_results = [r for r in topic_results.values() if not r["is_valid"]]
        is_valid = len(failed_results) == 0
        if is_valid:
            overall_message = (
                f"Hz validation PASSED: all topics meet {target_hz} Hz target"
            )
        else:
            failed_count = len(failed_results)
            overall_message = (
                f"Hz validation FAILED: {failed_count}/{len(topic_results)} topics "
                f"below {target_hz * min_ratio:.1f} Hz minimum"
            )
    else:
        is_valid = timing_source_valid
        if is_valid:
            overall_message = (
                f"Hz validation PASSED: timing source '{timing_source}' "
                f"meets {target_hz} Hz target"
            )
        else:
            overall_message = (
                f"Hz validation FAILED: timing source '{timing_source}' "
                f"{timing_source_result['actual_hz']:.1f} Hz < "
                f"{timing_source_result['min_hz']:.1f} Hz"
            )

    return is_valid, overall_message, timing_source_valid


def validate_from_timestamps(
    timestamps: dict[str, list[int]],
    target_hz: float,
    min_ratio: float = 0.7,
    validate_all_topics: bool = False,
    camera_names: list[str] | None = None,
) -> HzValidationResult:
    """Validate Hz from pre-collected timestamps."""
    timing_source = camera_names[0] if camera_names else "observation"

    topic_results: dict[str, dict[str, Any]] = {}
    for topic, ts_list in timestamps.items():
        topic_results[topic] = _validate_topic_hz(
            ts_list, target_hz, min_ratio, topic
        )
    is_valid, overall_message, timing_source_valid = _build_overall_result(
        topic_results=topic_results,
        timing_source=timing_source,
        target_hz=target_hz,
        min_ratio=min_ratio,
        validate_all_topics=validate_all_topics,
    )

    return HzValidationResult(
        is_valid=is_valid,
        overall_message=overall_message,
        topic_results=topic_results,
        timing_source=timing_source,
        timing_source_valid=timing_source_valid,
    )
