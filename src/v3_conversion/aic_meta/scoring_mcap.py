"""Per-episode extractors for /scoring/* MCAP topics."""

from pathlib import Path
from typing import Any, Dict

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
