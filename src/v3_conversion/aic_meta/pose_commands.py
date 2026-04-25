"""Sparse extractor for /aic_controller/pose_commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory


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
                        continue
                    if record.channel_id not in decoders:
                        try:
                            decoders[record.channel_id] = (
                                decoder_factory.decoder_for(
                                    channel.message_encoding, schema
                                )
                            )
                        except Exception:
                            decoders[record.channel_id] = None
                            continue
                    decoder = decoders[record.channel_id]
                    if decoder is None:
                        continue
                    try:
                        msg = decoder(record.data)
                        t_ns = int(record.log_time)
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
                        continue
    except Exception:
        return rows
    return rows
