"""pyarrow schemas for meta/aic/*.parquet."""

import pyarrow as pa


TASK_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("run_folder", pa.string()),
    pa.field("trial_key", pa.string()),
    pa.field("trial_score_folder", pa.string()),
    pa.field("schema_version", pa.string()),
    pa.field("cable_type", pa.string()),
    pa.field("cable_name", pa.string()),
    pa.field("plug_type", pa.string()),
    pa.field("plug_name", pa.string()),
    pa.field("port_type", pa.string()),
    pa.field("port_name", pa.string()),
    pa.field("target_module", pa.string()),
    pa.field("success", pa.bool_()),
    pa.field("early_terminated", pa.bool_()),
    pa.field("early_term_source", pa.string()),
    pa.field("duration_sec", pa.float32()),
    pa.field("num_steps", pa.int32()),
    pa.field("policy", pa.string()),
    pa.field("seed", pa.int32()),
    pa.field("insertion_event_fired", pa.bool_()),
    pa.field("insertion_event_target", pa.string()),
    pa.field("insertion_event_time_sec", pa.float32()),
])


_CATEGORY_COLUMNS = [
    "contacts",
    "duration",
    "insertion_force",
    "traj_efficiency",
    "traj_smoothness",
]


SCORING_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("score_total", pa.float32()),
    pa.field("score_tier1", pa.float32()),
    pa.field("score_tier2", pa.float32()),
    pa.field("score_tier3", pa.float32()),
    *[
        pa.field(f"score_{name}", pa.float32())
        for name in _CATEGORY_COLUMNS
    ],
    *[
        pa.field(f"score_{name}_message", pa.string())
        for name in _CATEGORY_COLUMNS
    ],
])
# Re-order to interleave score / message as the spec documents
SCORING_SCHEMA = pa.schema(
    [
        SCORING_SCHEMA.field("episode_index"),
        SCORING_SCHEMA.field("score_total"),
        SCORING_SCHEMA.field("score_tier1"),
        SCORING_SCHEMA.field("score_tier2"),
        SCORING_SCHEMA.field("score_tier3"),
        *sum(
            (
                [
                    SCORING_SCHEMA.field(f"score_{name}"),
                    SCORING_SCHEMA.field(f"score_{name}_message"),
                ]
                for name in _CATEGORY_COLUMNS
            ),
            [],
        ),
    ]
)


_RAIL_STRUCT = pa.struct([
    pa.field("name", pa.string()),
    pa.field("entity_present", pa.bool_()),
    pa.field("entity_name", pa.string()),
])


SCENE_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("plug_port_distance_init", pa.float32()),
    pa.field(
        "initial_plug_pose_rel_gripper",
        pa.list_(pa.float32(), list_size=6),
    ),
    pa.field("scene_rails", pa.list_(_RAIL_STRUCT)),
])


_FRAME_STRUCT = pa.struct([
    pa.field("frame_id", pa.string()),
    pa.field("parent_frame_id", pa.string()),
    pa.field("pose", pa.list_(pa.float32(), list_size=7)),
])


TF_SNAPSHOTS_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("scoring_frames_initial", pa.list_(_FRAME_STRUCT)),
    pa.field("scoring_frames_final", pa.list_(_FRAME_STRUCT)),
])


POSE_COMMANDS_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("t_ns", pa.int64(), nullable=False),
    pa.field("time_sec", pa.float32()),
    pa.field("pose", pa.list_(pa.float32(), list_size=7)),
    pa.field("velocity", pa.list_(pa.float32(), list_size=6)),
    pa.field("stiffness", pa.list_(pa.float32())),
    pa.field("damping", pa.list_(pa.float32())),
])
