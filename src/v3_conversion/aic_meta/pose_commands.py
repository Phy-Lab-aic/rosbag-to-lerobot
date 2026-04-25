"""Sparse extractor for /aic_controller/pose_commands."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List

from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory


logger = logging.getLogger(__name__)


def _pose_list(pose: Any) -> list[float]:
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def _twist_list(twist: Any) -> list[float]:
    return [
        float(twist.linear.x),
        float(twist.linear.y),
        float(twist.linear.z),
        float(twist.angular.x),
        float(twist.angular.y),
        float(twist.angular.z),
    ]


def extract_pose_commands(
    bag_path: Path,
    episode_index: int,
    episode_start_ns: int,
) -> List[Dict[str, Any]]:
    decoder_factory = DecoderFactory()
    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    decoders: dict[int, Any] = {}
    rows: list[Dict[str, Any]] = []
    message_count = 0

    try:
        with Path(bag_path).open("rb") as f:
            for record in StreamReader(f, record_size_limit=None).records:
                record_type = type(record).__name__
                if record_type == "Schema":
                    schemas[record.id] = record
                elif record_type == "Channel":
                    channels[record.id] = record
                elif record_type == "Message":
                    channel = channels.get(record.channel_id)
                    if (
                        channel is None
                        or channel.topic != "/aic_controller/pose_commands"
                    ):
                        continue
                    schema = schemas.get(channel.schema_id)
                    if schema is None:
                        logger.warning(
                            "Skipping pose command on channel %s: missing schema id %s",
                            record.channel_id,
                            channel.schema_id,
                        )
                        continue
                    if record.channel_id not in decoders:
                        try:
                            decoder = decoder_factory.decoder_for(
                                channel.message_encoding, schema
                            )
                            if decoder is None:
                                raise ValueError("decoder factory returned None")
                            decoders[record.channel_id] = decoder
                        except Exception:
                            decoders[record.channel_id] = None
                            logger.warning(
                                "Unable to construct decoder for pose commands "
                                "(topic=%s, channel=%s, schema=%s): %s",
                                channel.topic,
                                record.channel_id,
                                getattr(schema, "name", channel.schema_id),
                                traceback.format_exc(limit=1).strip(),
                            )
                            continue
                    decoder = decoders[record.channel_id]
                    if decoder is None:
                        continue
                    try:
                        msg = decoder(record.data)
                        message_count += 1
                    except Exception:
                        logger.warning(
                            "Unable to decode pose command "
                            "(topic=%s, channel=%s, schema=%s, log_time=%s): %s",
                            channel.topic,
                            record.channel_id,
                            getattr(schema, "name", channel.schema_id),
                            record.log_time,
                            traceback.format_exc(limit=1).strip(),
                        )
                        continue
                    t_ns = int(record.log_time)
                    try:
                        rows.append(
                            {
                                "episode_index": int(episode_index),
                                "t_ns": t_ns,
                                "time_sec": float(
                                    (t_ns - episode_start_ns) / 1_000_000_000
                                ),
                                "pose": _pose_list(msg.pose),
                                "velocity": _twist_list(msg.velocity),
                                "stiffness": [
                                    float(x)
                                    for x in getattr(msg, "target_stiffness", [])
                                ],
                                "damping": [
                                    float(x)
                                    for x in getattr(msg, "target_damping", [])
                                ],
                            }
                        )
                    except Exception:
                        logger.warning(
                            "Skipping malformed pose command "
                            "(topic=%s, channel=%s, schema=%s, log_time=%s): %s",
                            channel.topic,
                            record.channel_id,
                            getattr(schema, "name", channel.schema_id),
                            record.log_time,
                            traceback.format_exc(limit=1).strip(),
                        )
                        continue
    except Exception:
        logger.warning(
            "Unable to read pose commands from %s after %d decoded messages "
            "and %d materialized rows: %s",
            bag_path,
            message_count,
            len(rows),
            traceback.format_exc(limit=1).strip(),
        )
        return rows
    return rows
