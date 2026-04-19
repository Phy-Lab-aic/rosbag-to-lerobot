"""Per-episode extractors for /scoring/* MCAP topics."""

from pathlib import Path
from typing import Any, Dict, List

from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory


_DEFAULT_RESULT = {
    "insertion_event_fired": False,
    "insertion_event_target": "",
    "insertion_event_time_sec": float("nan"),
}


def extract_insertion_event(
    bag_path: Path, episode_start_ns: int
) -> Dict[str, Any]:
    """Scan /scoring/insertion_event; return first-message target + relative time."""
    factory = DecoderFactory()
    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    decoders: dict[int, Any] = {}

    try:
        with open(bag_path, "rb") as f:
            for record in StreamReader(f, record_size_limit=None).records:
                rtype = type(record).__name__
                if rtype == "Schema":
                    schemas[record.id] = record
                elif rtype == "Channel":
                    channels[record.id] = record
                elif rtype == "Message":
                    ch = channels.get(record.channel_id)
                    if not ch or ch.topic != "/scoring/insertion_event":
                        continue
                    sc = schemas.get(ch.schema_id)
                    if not sc:
                        continue
                    if record.channel_id not in decoders:
                        decoders[record.channel_id] = factory.decoder_for(
                            ch.message_encoding, sc
                        )
                    dec = decoders.get(record.channel_id)
                    if dec is None:
                        continue
                    try:
                        msg = dec(record.data)
                    except Exception:
                        continue
                    return {
                        "insertion_event_fired": True,
                        "insertion_event_target": str(getattr(msg, "data", "")),
                        "insertion_event_time_sec": (
                            float(record.log_time - episode_start_ns) / 1e9
                        ),
                    }
    except Exception:
        return dict(_DEFAULT_RESULT)

    return dict(_DEFAULT_RESULT)


def _collect_tf_messages(bag_path: Path):
    """Yield ``(log_time_ns, msg)`` for every ``/scoring/tf`` message."""
    factory = DecoderFactory()
    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    decoders: dict[int, Any] = {}

    with open(bag_path, "rb") as f:
        for record in StreamReader(f, record_size_limit=None).records:
            rtype = type(record).__name__
            if rtype == "Schema":
                schemas[record.id] = record
            elif rtype == "Channel":
                channels[record.id] = record
            elif rtype == "Message":
                ch = channels.get(record.channel_id)
                if not ch or ch.topic != "/scoring/tf":
                    continue
                sc = schemas.get(ch.schema_id)
                if not sc:
                    continue
                if record.channel_id not in decoders:
                    decoders[record.channel_id] = factory.decoder_for(
                        ch.message_encoding, sc
                    )
                dec = decoders.get(record.channel_id)
                if dec is None:
                    continue
                yield record.log_time, dec(record.data)


def _transform_to_row(transform) -> Dict[str, Any]:
    t = transform.transform.translation
    q = transform.transform.rotation
    return {
        "frame_id": str(transform.child_frame_id),
        "parent_frame_id": str(transform.header.frame_id),
        "pose": [
            float(t.x),
            float(t.y),
            float(t.z),
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
        ],
    }


def extract_scoring_tf_snapshots(
    bag_path: Path, window_ns: int = 1_000_000_000
) -> Dict[str, Any]:
    """Assemble initial/final snapshots from ``/scoring/tf`` traffic."""
    messages: List[tuple[int, Any]] = list(_collect_tf_messages(bag_path))
    if not messages:
        return {"scoring_frames_initial": [], "scoring_frames_final": []}

    first_t = messages[0][0]
    last_t = messages[-1][0]
    initial_cutoff = first_t + window_ns
    final_cutoff = last_t - window_ns

    initial_map: Dict[str, Dict[str, Any]] = {}
    final_map: Dict[str, Dict[str, Any]] = {}

    for t_ns, msg in messages:
        for transform in msg.transforms:
            row = _transform_to_row(transform)
            if t_ns <= initial_cutoff:
                initial_map[row["frame_id"]] = row
            if t_ns >= final_cutoff:
                final_map[row["frame_id"]] = row

    return {
        "scoring_frames_initial": list(initial_map.values()),
        "scoring_frames_final": list(final_map.values()),
    }
