import pyarrow as pa

from v3_conversion.aic_meta import schemas


def test_task_schema_has_required_columns():
    required = {
        "episode_index", "run_folder", "trial_key", "trial_score_folder",
        "schema_version",
        "cable_type", "cable_name", "plug_type", "plug_name",
        "port_type", "port_name", "target_module",
        "success", "early_terminated", "early_term_source",
        "duration_sec", "num_steps",
        "policy", "seed",
        "insertion_event_fired",
        "insertion_event_target",
        "insertion_event_time_sec",
    }
    assert set(schemas.TASK_SCHEMA.names) == required
    assert schemas.TASK_SCHEMA.field("episode_index").type == pa.int32()


def test_scoring_schema_includes_all_categories():
    required = {
        "episode_index",
        "score_total",
        "score_tier1", "score_tier2", "score_tier3",
        "score_contacts", "score_contacts_message",
        "score_duration", "score_duration_message",
        "score_insertion_force", "score_insertion_force_message",
        "score_traj_efficiency", "score_traj_efficiency_message",
        "score_traj_smoothness", "score_traj_smoothness_message",
    }
    assert set(schemas.SCORING_SCHEMA.names) == required


def test_scene_schema_has_nested_rails():
    f = schemas.SCENE_SCHEMA.field("scene_rails")
    inner = f.type.value_type  # struct<...>
    inner_fields = {inner.field(i).name for i in range(inner.num_fields)}
    assert inner_fields == {"name", "entity_present", "entity_name"}


def test_tf_snapshots_schema_poses_are_float32_lists():
    f = schemas.TF_SNAPSHOTS_SCHEMA.field("scoring_frames_initial")
    struct_type = f.type.value_type
    pose_field = next(
        struct_type.field(i) for i in range(struct_type.num_fields)
        if struct_type.field(i).name == "pose"
    )
    assert pose_field.type.value_type == pa.float32()
