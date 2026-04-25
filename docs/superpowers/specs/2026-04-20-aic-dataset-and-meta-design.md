# AIC Dataset + Curation Meta Design (v1)

- **Date:** 2026-04-20
- **Status:** Draft, awaiting user review
- **Source dataset:** `/home/weed/aic_community_e2e/` (run-level AIC benchmark dumps)
- **Target dataset:** LeRobot v3 dataset (`OUTPUT_PATH/<task_name>/`)
- **Author:** brainstorming session, captured for the writing-plans stage

## 1. Goals and scope

### 1.1 What this spec defines

1. A per-frame feature schema (`data/*.parquet` + `videos/`) that:
   - Uses `/joint_states` as the single source for both `observation.state` (`q_t`) and `action` (`q_{t+1}`), aligned with pi0 / pi0.5 / ACT / Diffusion Policy conventions.
   - Adds `observation.wrench` as a new 6-D feature sourced from `/fts_broadcaster/wrench`.
   - Records three raw camera image streams (`sensor_msgs/Image`, NOT `CompressedImage`) at a fixed 20 Hz grid.
2. A per-episode **curation-only** metadata layout under `meta/aic/` split into four thematic parquet files:
   - `meta/aic/task.parquet`
   - `meta/aic/scoring.parquet`
   - `meta/aic/scene.parquet`
   - `meta/aic/tf_snapshots.parquet`
3. Extraction rules from the source AIC run layout (`run_XX_*/...`) to the target dataset.
4. Auto-generation of the LeRobot native `task` string per episode.

### 1.2 What this spec does NOT define

- Concrete Python/Arrow code, function signatures, or module layout for the converter. Those belong to the implementation plan produced by `superpowers:writing-plans`.
- The pi0.5 training loop, dataloader config, or checkpoint handling. Only the dataset-side contract is in scope.
- The `mcap_reader` rewrite details needed to support the new image topics and action shift; only the requirements are listed.

### 1.3 Consumers

| Consumer | Reads |
|---|---|
| pi0.5 / ACT / Diffusion Policy trainer | `data/*.parquet`, `videos/*.mp4`, `meta/info.json`, `meta/stats.json`, `meta/tasks.jsonl`, `meta/episodes/*.parquet` |
| Data curation / analysis tooling | `meta/aic/*.parquet` (joined on `episode_index`) |
| Online policy at inference | Never reads the dataset; receives `joint_state`, `camera image`, `wrench`, and task info from the runtime |

The AIC curation meta (`meta/aic/*.parquet`) is **never** consumed by the model.

## 2. Source data layout

A single AIC run produces:

```
run_XX_YYYYMMDD_hhmmss/
├── config.yaml              # used: trials.trial_N.scene (rails + cable init pose)
├── policy.txt               # used: policy name
├── seed.txt                 # used: seed
├── scoring_run.yaml         # IGNORED (fully covered by scoring.yaml)
├── validation.json          # IGNORED (file-presence check only)
└── trial_N_scoreM/
    ├── scoring.yaml         # used: tier1/2/3 + 5 categories (+ messages)
    ├── tags.json            # used: schema_version, success, early_term_*
    ├── bag/
    │   ├── bag_trial_N_*.mcap   # primary data source; see §2.1
    │   └── metadata.yaml        # IGNORED (bag manifest)
    └── episode/
        ├── metadata.json    # used: semantic + outcome + plug_port_distance_init
        └── *.npy, images/   # IGNORED (replaced by MCAP extraction; see §2.2)
```

Each run produces **one episode** in the target dataset (one `trial_N_scoreM` per run in the current AIC workflow).

### 2.1 MCAP topics consumed

Future MCAP bags (post-recording update) must contain all of the following at or above 20 Hz:

| Topic | Schema | Use |
|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | `observation.state` (q_t), `action` (q_{t+1}) |
| `/fts_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` | `observation.wrench` |
| `/left_camera/image` | `sensor_msgs/msg/Image` (raw) | `observation.images.cam_left` |
| `/center_camera/image` | `sensor_msgs/msg/Image` (raw) | `observation.images.cam_center` |
| `/right_camera/image` | `sensor_msgs/msg/Image` (raw) | `observation.images.cam_right` |
| `/scoring/insertion_event` | `std_msgs/msg/String` | curation meta only |
| `/scoring/tf` | `tf2_msgs/msg/TFMessage` | curation meta only |

**Explicitly ignored topics** (present in MCAP but unused):
`/aic_controller/pose_commands`, `/aic_controller/controller_state`, `/aic_controller/joint_commands`, `/tf`, `/tf_static`, `/left_camera/camera_info`, `/center_camera/camera_info`, `/right_camera/camera_info`, `/left_camera/image/compressed`, `/center_camera/image/compressed`, `/right_camera/image/compressed`.

### 2.2 Why the `episode/` sidecar is ignored

`episode/*.npy` and `episode/images/*.png` duplicate information that will live in the new MCAPs once the recording pipeline is updated to publish raw camera images at 20 Hz. Relying on MCAP-only input keeps the converter a single data path and removes the PNG-from-disk fallback.

## 3. Target dataset layout (LeRobot v3)

```
<OUTPUT_PATH>/<task_name>/
├── meta/
│   ├── info.json                # LeRobot standard
│   ├── stats.json               # LeRobot standard
│   ├── tasks.jsonl              # LeRobot standard, auto-generated
│   ├── episodes/                # LeRobot standard (length, video offsets, ...)
│   │   └── chunk-000/file-000.parquet
│   └── aic/                     # NEW — curation-only, this spec
│       ├── task.parquet
│       ├── scoring.parquet
│       ├── scene.parquet
│       └── tf_snapshots.parquet
├── data/
│   └── chunk-000/file-000.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.cam_left/file-000.mp4
        ├── observation.images.cam_center/file-000.mp4
        └── observation.images.cam_right/file-000.mp4
```

- LeRobot native files are untouched in layout.
- All custom curation state lives under `meta/aic/`; every file is keyed by `episode_index` and joinable via `pandas.merge`.
- `meta/episodes/*.parquet` custom metadata columns from the current converter (`Serial_number`, `tags`, `grade`) are dropped in favour of the richer `meta/aic/task.parquet`; see §10.

## 4. Per-frame feature schema

### 4.1 Canonical 7-joint order

```
[
  shoulder_pan_joint,
  shoulder_lift_joint,
  elbow_joint,
  wrist_1_joint,
  wrist_2_joint,
  wrist_3_joint,
  gripper/left_finger_joint
]
```

`/joint_states.name` arrives in alphabetical order; every converter step MUST reorder to the canonical list above before writing `observation.state` or `action`.

### 4.2 Features and shapes

```
observation.state              float32 (7,)    q_t, absolute joint positions
observation.wrench             float32 (6,)    [Fx, Fy, Fz, Tx, Ty, Tz]  at tool_link
observation.images.cam_left    video (H, W, 3) RGB, from /left_camera/image
observation.images.cam_center  video (H, W, 3)
observation.images.cam_right   video (H, W, 3)
action                         float32 (7,)    q_{t+1}, absolute joint positions
task                           string          per-episode auto-generated; see §7
```

Image height/width are inherited from the `sensor_msgs/msg/Image` payload (expected 1152×1024 in the current sim, not hard-coded in the spec).

### 4.3 Timing grid

- **Target FPS: 20 Hz**, fixed. Matches the camera publish rate in the updated recording pipeline.
- Camera timestamps drive the grid. For every camera frame `t_k`:
  - `observation.state` / `action` use the nearest-before `/joint_states` sample.
  - `observation.wrench` uses the nearest-before `/fts_broadcaster/wrench` sample.
- If any sensor has no prior sample at `t_k` (warm-up region), drop that frame from the episode.

### 4.4 Action 1-step shift

- `action[t] = q_{t+1}` (absolute, not delta).
- `action[t] == q_t` is forbidden — it makes copying the observation the optimal policy.
- The last valid frame of an episode has no `q_{t+1}`; the episode length is reduced to `length = T - 1` and the final raw camera frame is dropped.
- Chunk horizon (`H`) is a dataloader hyperparameter (`H = 20` corresponds to 1 s at 20 Hz). The parquet stores only single-step actions; chunking happens at training time.

### 4.5 Normalization and pretraining alignment

- `observation.state` and `action` inhabit the same absolute joint-position space → pi0.5 UR5e `state_mean/std` and `action_mean/std` can be reused without recomputation.
- `observation.wrench` needs fresh statistics since pi0.5 UR5e pretraining does not include wrench; `meta/stats.json` must include wrench mean/std/min/max computed from this dataset.
- Gripper values stay continuous (position, not binary open/close). Deploy-time dispatch of the 7th action element to a separate gripper client is an inference-time concern and out of scope.

## 5. Per-episode curation meta (`meta/aic/*.parquet`)

Every file below has `episode_index: int32` as primary key. Joins across files use `pandas.merge(on="episode_index")`.

### 5.1 `meta/aic/task.parquet`

Identity, task semantics, outcome, run provenance, and the insertion-event extract in one row per episode.

| Column | Arrow type | Source | Notes |
|---|---|---|---|
| `episode_index` | `int32` | converter | PK, unique |
| `run_folder` | `string` | path | e.g. `"run_01_20260412_141241"` |
| `trial_key` | `string` | path | e.g. `"trial_1"` |
| `trial_score_folder` | `string` | path | e.g. `"trial_1_score95"` |
| `schema_version` | `string` | `tags.json.schema_version` | |
| `cable_type` | `string` | `episode/metadata.json.cable_type` | primary source |
| `cable_name` | `string` | `episode/metadata.json.cable_name` | |
| `plug_type` | `string` | `episode/metadata.json.plug_type` | |
| `plug_name` | `string` | `episode/metadata.json.plug_name` | |
| `port_type` | `string` | `episode/metadata.json.port_type` | |
| `port_name` | `string` | `episode/metadata.json.port_name` | |
| `target_module` | `string` | `episode/metadata.json.target_module` | |
| `success` | `bool` | `episode/metadata.json.success` | |
| `early_terminated` | `bool` | `episode/metadata.json.early_terminated` | |
| `early_term_source` | `string` | `episode/metadata.json.early_term_source` | `""` when not terminated |
| `duration_sec` | `float32` | `episode/metadata.json.duration_sec` | |
| `num_steps` | `int32` | `episode/metadata.json.num_steps` | raw policy-step count from the source; the LeRobot episode length may differ after 20 Hz resync and the final-frame drop |
| `policy` | `string` | `policy.txt` | |
| `seed` | `int32` | `seed.txt` | |
| `insertion_event_fired` | `bool` | MCAP `/scoring/insertion_event` | `true` iff at least one message observed |
| `insertion_event_target` | `string` | MCAP `/scoring/insertion_event` | last message's `data`; `""` when not fired |
| `insertion_event_time_sec` | `float32` | MCAP | seconds from episode start; `NaN` when not fired |

### 5.2 `meta/aic/scoring.parquet`

Tiered and categorised scoring with the human-readable messages that contain numeric detail (e.g. `"end-effector path length: 0.16 m"`).

| Column | Arrow type | Source |
|---|---|---|
| `episode_index` | `int32` | PK/FK |
| `score_total` | `float32` | `scoring.yaml.trial_N.total` |
| `score_tier1` | `float32` | `scoring.yaml.trial_N.tier_1.score` |
| `score_tier2` | `float32` | `scoring.yaml.trial_N.tier_2.score` |
| `score_tier3` | `float32` | `scoring.yaml.trial_N.tier_3.score` |
| `score_contacts` | `float32` | `tier_2.categories.contacts.score` |
| `score_contacts_message` | `string` | `tier_2.categories.contacts.message` |
| `score_duration` | `float32` | `tier_2.categories.duration.score` |
| `score_duration_message` | `string` | `tier_2.categories.duration.message` |
| `score_insertion_force` | `float32` | `tier_2.categories.insertion force.score` |
| `score_insertion_force_message` | `string` | `tier_2.categories.insertion force.message` |
| `score_traj_efficiency` | `float32` | `tier_2.categories.trajectory efficiency.score` |
| `score_traj_efficiency_message` | `string` | `tier_2.categories.trajectory efficiency.message` |
| `score_traj_smoothness` | `float32` | `tier_2.categories.trajectory smoothness.score` |
| `score_traj_smoothness_message` | `string` | `tier_2.categories.trajectory smoothness.message` |

Tier-level messages (`score_tier{1,2,3}_message`) are intentionally dropped — the numeric score itself already conveys success/fail at that granularity.

### 5.3 `meta/aic/scene.parquet`

Static scene setup per episode. Numerical placements of present entities live in `tf_snapshots.parquet`; here we keep only the slot → entity mapping that encodes which slots are occupied vs empty (information `/scoring/tf` cannot convey, since empty slots have no frame).

| Column | Arrow type | Source |
|---|---|---|
| `episode_index` | `int32` | PK/FK |
| `plug_port_distance_init` | `float32` | `episode/metadata.json.plug_port_distance` |
| `initial_plug_pose_rel_gripper` | `list<float32>[6]` | `config.yaml.trials.trial_N.scene.cables.cable_0.pose` — `[gripper_offset.x, y, z, roll, pitch, yaw]` |
| `scene_rails` | `list<struct{name, entity_present, entity_name}>` | `config.yaml.trials.trial_N.scene.task_board.*_rail*` |

`scene_rails` element schema:
```
struct{
  name:           string,   # slot name (e.g. "nic_rail_0", "sc_mount_rail_1")
  entity_present: bool,
  entity_name:    string    # "" when absent
}
```

The list MUST include every rail slot listed under `task_board` in the source `config.yaml`, in the source order.

### 5.4 `meta/aic/tf_snapshots.parquet`

Ground-truth pose snapshots derived from `/scoring/tf`. The topic broadcasts individual transforms incrementally; we assemble snapshots as follows:

- **Initial window**: the first 1.0 s of `/scoring/tf` traffic, measured from the MCAP `log_time` of the first `/scoring/tf` message. For each unique `child_frame_id` seen in this window, keep the most recent transform.
- **Final window**: the last 1.0 s of `/scoring/tf` traffic, measured backwards from the `log_time` of the last `/scoring/tf` message. Same per-frame rule.
- If fewer than 1.0 s of traffic exists (very short episode), collapse the window to whatever is available.

| Column | Arrow type | Notes |
|---|---|---|
| `episode_index` | `int32` | PK/FK |
| `scoring_frames_initial` | `list<struct>` | see element schema below |
| `scoring_frames_final` | `list<struct>` | same schema |

Element schema:
```
struct{
  frame_id:        string,         # child_frame_id (e.g. "sfp_tip", "sfp_port_0")
  parent_frame_id: string,         # header.frame_id
  pose:            list<float32>[7] # [x, y, z, qx, qy, qz, qw]
}
```

Quaternion convention: `[qx, qy, qz, qw]`, matching ROS 2 `geometry_msgs/Quaternion` member order.

If `/scoring/tf` is missing or silent for an episode, both columns are written as empty lists (`[]`), not null.

## 6. Scoring topic extracts

### 6.1 `/scoring/insertion_event`

- Feeds only the three `insertion_event_*` columns in `task.parquet`.
- `std_msgs/String` carries no `header`, so `insertion_event_time_sec` is computed from the MCAP **record** `log_time` of the first observed message on this topic, minus the `log_time` of the camera frame kept as `frame_index == 0`.
- If the topic fires more than once within a single episode (not expected under current scoring rules), `insertion_event_target` and `insertion_event_time_sec` reflect the **first** message; a warning is logged.
- The model never observes this signal, avoiding test-time leakage (the inference runtime does not emit `/scoring/*`).

### 6.2 `/scoring/tf`

- Feeds `scoring_frames_initial` and `scoring_frames_final` in `tf_snapshots.parquet` only.
- Never feeds `data/*.parquet` or `videos/*.mp4`.

## 7. Language (`task`) generation

Per-episode string auto-generated from semantic fields, then passed to `LeRobotDataset.add_frame(task=...)`. LeRobot de-duplicates unique strings into `meta/tasks.jsonl`; every frame stores only `task_index`.

### 7.1 Template

```
"Insert the {cable_type_readable} cable's {plug_name} tip into {port_name} on {target_module}."
```

### 7.2 `cable_type_readable` mapping

| Source value | Readable form |
|---|---|
| `sfp_sc` | `"SFP-to-SC"` |
| `sc_sfp` | `"SC-to-SFP"` |
| `sfp` | `"SFP"` |
| `sc` | `"SC"` |
| `lc` | `"LC"` |
| other | original value unchanged |

### 7.3 Fallback

If any of `cable_type`, `plug_name`, `port_name`, `target_module` is missing, emit `"Insert cable."` as the task string and log a warning.

### 7.4 Non-duplication

The task string is stored in LeRobot's native `meta/tasks.jsonl` only. It is **not** stored as a column in `meta/aic/task.parquet` — regenerate from semantic fields if needed downstream.

## 8. Ignored sources and non-goals

- `scoring_run.yaml`: redundant with `scoring.yaml`.
- `validation.json`: file-presence audit only; no semantic value.
- `config.yaml.task_board_limits`: constant across the benchmark; not per-episode.
- `config.yaml.robot.home_joint_positions`: not needed (user directive).
- `config.yaml.scoring.topics`: describes recording spec, not per-episode state.
- `episode/*.npy`, `episode/images/*`: replaced by MCAP extraction.
- `/aic_controller/*` topics: inference does not provide these.
- `/tf`, `/tf_static`: robot state is captured via `/joint_states`; on-board TF is not used.
- `/*_camera/camera_info`: camera intrinsics are constant across the dataset; persisting them is deferred.
- CompressedImage variants of camera topics.
- Tier-level scoring messages (`tier1/2/3_message`): their numeric score already encodes pass/fail.
- `trial_duration_sec` (`tags.json`): redundant with `duration_sec` (`metadata.json`) within the expected tolerance.

## 9. Implementation touch points

These are **requirements for the implementation plan**, not code. Addressed in writing-plans output.

1. **`mcap_reader.extract_frames`**: introduce a camera-timestamp-driven sync loop; feed `/joint_states`, `/fts_broadcaster/wrench`, and three raw `sensor_msgs/Image` streams through the existing schema-dispatch path.
2. **Action shift**: produce `action[t] = q_{t+1}` (not `q_t`); drop the final frame of every episode.
3. **`data_converter.build_frame`**: stop emitting a shared `leader_msgs` clone from `/joint_states`; new post-processing step performs the shift once per episode.
4. **`data_creator.create_dataset`**: register a new `observation.wrench` feature (`float32`, shape `(6,)`, names `["Fx","Fy","Fz","Tx","Ty","Tz"]`) alongside `observation.state`.
5. **New module `aic_meta.py` (or equivalent)**: loads `config.yaml` / `tags.json` / `scoring.yaml` / `episode/metadata.json` / `policy.txt` / `seed.txt`, parses `/scoring/*` topics from the same MCAP, and writes the four `meta/aic/*.parquet` files at the end of the conversion run.
6. **`task` auto-generation**: helper that takes semantic fields and produces the LeRobot task string; fed to `add_frame`.
7. **Remove PNG-from-disk path**: no code path reads `episode/images/*.png`.
8. **Remove `pose_commands` action path**: update any config loader / validator that still expects `/aic_controller/pose_commands` or similar.
9. **Existing `patch_episodes_metadata` custom fields** (`Serial_number`, `tags`, `grade`): stop writing them; all traceability is handled by `meta/aic/task.parquet`. Tests that assert their presence need updating.

## 10. Migration / backward compatibility

- **`meta/aic.parquet` (single-file, current)** → removed. Replaced by `meta/aic/*.parquet` (directory). Existing readers that hard-code the single file must be updated; column coverage is a strict superset of the old file.
- **`meta/episodes/*.parquet` custom columns** (`Serial_number`, `tags`, `grade`): dropped. `run_folder` in `meta/aic/task.parquet` provides the same traceability.
- **`config.json` schema** (project-level converter config): `action_topics_map` behaviour changes — `"leader": "/joint_states"` now triggers a 1-step shift rather than a synchronous copy. Existing configs that relied on the zero-shift behaviour must be reviewed.
- **Existing episodes in a merged dataset**: when running in `--merge` mode, new episodes get `meta/aic/*.parquet` rows; pre-existing episodes without source files in `INPUT_PATH` cannot be back-filled from the merge alone.

## 11. Open questions

None as of 2026-04-20. All design choices were confirmed during the brainstorming session:

| Decision | Resolution |
|---|---|
| Meta consumer | curation-only (option A) |
| Field scope | tier 3 |
| Storage layout | flat columns, split into 4 thematic files (option B revision) |
| Language generation | template auto-gen (option ii) |
| FPS | 20 Hz fixed, camera-grid |
| Image source | raw `sensor_msgs/Image` in MCAP |
| Action space | future joint positions, absolute, 1-step shift |
| Home joint positions | not recorded |
| Camera info | not recorded |
| Scoring topics | per-episode snapshots into `aic/task.parquet` and `aic/tf_snapshots.parquet` |

Future work (tracked outside this spec):
- Revisit `camera_info` persistence once multi-camera calibration matters for downstream 3D.
- Revisit whether `observation.joint_velocity` / `observation.joint_effort` should become default features once empirical gains are measured.

## Appendix A — Worked example

Source: `/home/weed/aic_community_e2e/run_01_20260412_141241/trial_1_score95`.

### A.1 `meta/aic/task.parquet` — row 0

```json
{
  "episode_index": 0,
  "run_folder": "run_01_20260412_141241",
  "trial_key": "trial_1",
  "trial_score_folder": "trial_1_score95",
  "schema_version": "0.1.0",
  "cable_type": "sfp_sc",
  "cable_name": "cable_0",
  "plug_type": "sfp",
  "plug_name": "sfp_tip",
  "port_type": "sfp",
  "port_name": "sfp_port_0",
  "target_module": "nic_card_mount_0",
  "success": true,
  "early_terminated": true,
  "early_term_source": "insertion_event",
  "duration_sec": 24.6827,
  "num_steps": 286,
  "policy": "cheatcode",
  "seed": 42,
  "insertion_event_fired": true,
  "insertion_event_target": "/nic_card_mount_0/sfp_port_0",
  "insertion_event_time_sec": 22.84
}
```

### A.2 `meta/aic/scoring.parquet` — row 0

```json
{
  "episode_index": 0,
  "score_total": 94.6802,
  "score_tier1": 1.0,
  "score_tier2": 18.6802,
  "score_tier3": 75.0,
  "score_contacts": 0.0,
  "score_contacts_message": "No contact detected.",
  "score_duration": 7.6294,
  "score_duration_message": "Task duration: 25.03 seconds.",
  "score_insertion_force": 0.0,
  "score_insertion_force_message": "No excessive force detected",
  "score_traj_efficiency": 5.8843,
  "score_traj_efficiency_message": "Total end-effector path length: 0.16 m, initial plug-port distance: 0.14 m",
  "score_traj_smoothness": 5.1665,
  "score_traj_smoothness_message": "Average linear jerk magnitude of the end effector: 6.95 m/s^3"
}
```

### A.3 `meta/aic/scene.parquet` — row 0

```json
{
  "episode_index": 0,
  "plug_port_distance_init": 0.001,
  "initial_plug_pose_rel_gripper": [0.0, 0.015385, 0.04245, 0.4432, -0.4838, 1.3303],
  "scene_rails": [
    {"name": "nic_rail_0",       "entity_present": true,  "entity_name": "nic_card_0"},
    {"name": "nic_rail_1",       "entity_present": false, "entity_name": ""},
    {"name": "nic_rail_2",       "entity_present": false, "entity_name": ""},
    {"name": "nic_rail_3",       "entity_present": false, "entity_name": ""},
    {"name": "nic_rail_4",       "entity_present": false, "entity_name": ""},
    {"name": "sc_rail_0",        "entity_present": true,  "entity_name": "sc_mount_0"},
    {"name": "sc_rail_1",        "entity_present": false, "entity_name": ""},
    {"name": "lc_mount_rail_0",  "entity_present": true,  "entity_name": "lc_mount_0"},
    {"name": "sfp_mount_rail_0", "entity_present": true,  "entity_name": "sfp_mount_0"},
    {"name": "sc_mount_rail_0",  "entity_present": true,  "entity_name": "sc_mount_0"},
    {"name": "lc_mount_rail_1",  "entity_present": true,  "entity_name": "lc_mount_1"},
    {"name": "sfp_mount_rail_1", "entity_present": false, "entity_name": ""},
    {"name": "sc_mount_rail_1",  "entity_present": false, "entity_name": ""}
  ]
}
```

### A.4 `meta/aic/tf_snapshots.parquet` — row 0 (illustrative)

Exact frame list is produced by MCAP parsing at conversion time; the entries below are representative only.

```json
{
  "episode_index": 0,
  "scoring_frames_initial": [
    {"frame_id": "task_board",       "parent_frame_id": "world",            "pose": [0.15, -0.20, 1.14, 0.0, 0.0, 0.99999, 0.00129]},
    {"frame_id": "nic_card_mount_0", "parent_frame_id": "task_board",       "pose": [-0.0025, 0.0, 0.02, 0.0, 0.0, 0.0796, 0.9968]},
    {"frame_id": "sfp_port_0",       "parent_frame_id": "nic_card_mount_0", "pose": [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0]},
    {"frame_id": "sfp_tip",          "parent_frame_id": "tool_link",        "pose": [0.0, 0.015385, 0.04245, 0.2036, -0.2364, 0.6108, 0.7241]},
    {"frame_id": "cable_0",          "parent_frame_id": "tool_link",        "pose": [0.0, 0.015385, 0.04245, 0.2036, -0.2364, 0.6108, 0.7241]}
  ],
  "scoring_frames_final": [
    {"frame_id": "task_board",       "parent_frame_id": "world",            "pose": [0.15, -0.20, 1.14, 0.0, 0.0, 0.99999, 0.00129]},
    {"frame_id": "nic_card_mount_0", "parent_frame_id": "task_board",       "pose": [-0.0025, 0.0, 0.02, 0.0, 0.0, 0.0796, 0.9968]},
    {"frame_id": "sfp_port_0",       "parent_frame_id": "nic_card_mount_0", "pose": [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0]},
    {"frame_id": "sfp_tip",          "parent_frame_id": "sfp_port_0",       "pose": [0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 1.0]},
    {"frame_id": "cable_0",          "parent_frame_id": "tool_link",        "pose": [0.0, 0.015385, 0.04245, 0.2036, -0.2364, 0.6108, 0.7241]}
  ]
}
```

### A.5 `data/chunk-000/file-000.parquet` — frame 0

```json
{
  "episode_index": 0,
  "frame_index": 0,
  "timestamp": 0.0,
  "task_index": 0,
  "observation.state":  [-0.15699, -1.35343, -1.69120, -1.66781, 1.57083, 1.41373, 0.00730],
  "observation.wrench": [0.00966, 0.15600, 20.64032, 0.10018, -0.12033, -0.08073],
  "action":             [-0.15683, -1.35365, -1.69125, -1.66744, 1.57083, 1.41390, 0.00730],
  "observation.images.cam_left":   "<video frame 0>",
  "observation.images.cam_center": "<video frame 0>",
  "observation.images.cam_right":  "<video frame 0>"
}
```

### A.6 `meta/tasks.jsonl` — resulting entry

```jsonl
{"task_index": 0, "task": "Insert the SFP-to-SC cable's sfp_tip into sfp_port_0 on nic_card_mount_0."}
```

## Sign-off

Approved 2026-04-20. Implementation plan: `docs/superpowers/plans/2026-04-20-aic-dataset-and-meta-converter.md`.
