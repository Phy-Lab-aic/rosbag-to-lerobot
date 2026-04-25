from pathlib import Path

from mcap.stream_reader import StreamReader


def _topics(path: Path) -> set[str]:
    topics: set[str] = set()
    channels = {}
    with path.open("rb") as f:
        for record in StreamReader(f, record_size_limit=None).records:
            record_type = type(record).__name__
            if record_type == "Channel":
                channels[record.id] = record.topic
                topics.add(record.topic)
            elif record_type == "Message":
                topic = channels.get(record.channel_id)
                if topic:
                    topics.add(topic)
    return topics


def test_build_mcap_fixture_writes_auxiliary_topics(build_mcap_fixture, tmp_path: Path):
    bag = build_mcap_fixture(
        path=tmp_path / "aux.mcap",
        joint_states=[
            (0, ["j0"], [0.1], [1.1]),
        ],
        controller_state=[
            (0, -0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0),
        ],
        tf=[
            (0, [("base_link", "tcp_link", 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)]),
        ],
        pose_commands=[
            (
                0,
                -0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0,
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                [90.0] * 36,
                [20.0] * 36,
            ),
        ],
    )

    assert {
        "/joint_states",
        "/aic_controller/controller_state",
        "/tf",
        "/aic_controller/pose_commands",
    }.issubset(_topics(bag))
