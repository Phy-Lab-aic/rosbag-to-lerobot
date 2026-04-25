# AIC Dataset + Curation Meta Design (v2)

- **Date:** 2026-04-26
- **Status:** Draft, awaiting user review
- **Supersedes:** `docs/superpowers/specs/2026-04-20-aic-dataset-and-meta-design.md`
- **Source dataset:** `/home/weed/aic_community_e2e/`
- **Target dataset:** LeRobot v3 dataset (`OUTPUT_PATH/<task_name>/`)

## 1. Purpose

This v2 design keeps the working LeRobot conversion path and narrows new work to high-value AIC extensions:

- Keep the 20 Hz camera-timestamp master clock.
- Keep `action = observation.state[t+1]` with the final raw frame dropped.
- Keep LeRobot-standard chunking and video/data layout.
- Keep `meta/aic/*.parquet` as the canonical AIC metadata format.
- Add per-frame `observation.velocity` and pose labels needed for vision pretraining and dense reward experiments.
- Preserve sparse controller commands as sparse metadata instead of expanding them onto every 20 Hz frame.

The main correction from v1 is that `/aic_controller/controller_state`, `/aic_controller/pose_commands`, `/tf`, and `/scoring/tf` are no longer treated as ignored. They are used selectively according to whether the data is per-frame, sparse, or episode-level.

## 2. Non-Goals

- Do not replace AIC parquet metadata with JSONL.
- Do not add `meta/aic/episode_info.parquet` in this iteration.
- Do not store precomputed weighted rewards.
- Do not store `reward.contact_force_norm`; derive it from `observation.wrench`.
- Do not add frame-level `success` or `violation` flags for imitation learning.
- Do not store raw 500 Hz `/scoring/tf` or 1000 Hz `/tf` streams.

## 3. Source Gating

Only convert a run when all required run-level checks pass:

- `validation.json` exists and indicates all checks passed.
- `episode/metadata.json` exists.
- Required LeRobot frame-stream topics exist: cameras, `/joint_states`, and any configured required wrench topic.

Skip, rather than repair, runs missing `episode/metadata.json`. That file is the canonical source for task semantics and episode outcome.

The following topics are optional and must not fail conversion when absent:

- `/aic/gazebo/contacts/off_limit`
- `/aic_controller/joint_commands`
- `/scoring/tf_static`

## 4. Per-Frame LeRobot Features

`data/*.parquet` contains only data that is directly useful per training step.

| Feature | Shape | Source | Notes |
|---|---:|---|---|
| `observation.state` | `(7,)` | `/joint_states.position` | Canonical joint order |
| `observation.velocity` | `(7,)` | `/joint_states.velocity` | Canonical joint order |
| `observation.wrench` | `(6,)` | `/fts_broadcaster/wrench` | `[Fx, Fy, Fz, Tx, Ty, Tz]` |
| `observation.images.{camera}` | `(H, W, 3)` | `/{left,center,right}_camera/image` | RGB video features |
| `action` | `(7,)` | `observation.state[t+1]` | Absolute next joint positions |
| `label.tcp_pose` | `(7,)` | Controller state, then `/tf` fallback | `[x, y, z, qx, qy, qz, qw]` in base frame |
| `label.plug_pose_base` | `(7,)` | `/scoring/tf` | Ground-truth plug pose |
| `label.port_pose_base` | `(7,)` | `/scoring/tf` | Ground-truth port pose |
| `label.target_module_pose_base` | `(7,)` | `/scoring/tf` | Ground-truth target module pose |
| `label.*_valid` | `bool` | converter | Mask for missing label poses |

All per-frame arrays use the same 20 Hz camera master clock. After building the raw frame stream, the converter applies the existing one-step action shift and trims all per-frame arrays to `T - 1`.

If a label pose is unavailable at a 20 Hz tick, the frame stays in the episode. The pose value is filled with `NaN`, and the corresponding `label.*_valid` flag is `false`.

## 5. Pose Label Extraction

### 5.1 TCP Pose

`label.tcp_pose` source priority:

1. `/aic_controller/controller_state.tcp_pose`
2. `/tf` transform-chain fallback

The controller state topic is preferred because the MCAP already contains FK-resolved TCP poses at high frequency. `/tf` fallback exists for datasets that do not record controller state, but it requires composing transforms directly from MCAP data without ROS runtime dependencies.

### 5.2 Plug, Port, and Target Module Poses

`label.plug_pose_base`, `label.port_pose_base`, and `label.target_module_pose_base` come from `/scoring/tf` sampled onto the 20 Hz camera grid.

The converter does not store raw `/scoring/tf`. It keeps only the pose labels needed by training, reward experiments, and analysis. Frame identities come from episode metadata:

- plug: `episode/metadata.json.plug_name`
- port: `episode/metadata.json.port_name`
- target module: `episode/metadata.json.target_module`

The implementation must normalize frame-name differences in one place, because source metadata and TF child frame IDs may use slightly different suffixes such as link names.

## 6. AIC Metadata Layout

`meta/aic` remains split by data responsibility. These files are the canonical metadata tables:

- `meta/aic/task.parquet`
- `meta/aic/scoring.parquet`
- `meta/aic/scene.parquet`
- `meta/aic/tf_snapshots.parquet`

Each table is keyed by `episode_index`. This split stays canonical because the tables have different column families and update reasons. A wide `episode_info.parquet` is intentionally not added.

## 7. Sparse Metadata

### 7.1 Pose Commands

`/aic_controller/pose_commands` must not be nearest-filled onto the 20 Hz frame stream. Expanding sparse commands would erase command timing.

Add `meta/aic/pose_commands.parquet`:

| Column | Type | Notes |
|---|---|---|
| `episode_index` | `int32` | FK to LeRobot episode |
| `t_ns` | `int64` | MCAP log time |
| `time_sec` | `float32` | Seconds from episode start |
| `pose` | `fixed_size_list<float32>[7]` | `[x, y, z, qx, qy, qz, qw]` |
| `velocity` | `fixed_size_list<float32>[6]` | Linear then angular velocity |
| `stiffness` | `list<float32>` | Raw matrix/list as recorded |
| `damping` | `list<float32>` | Raw matrix/list as recorded |

### 7.2 Events

`meta/aic/events.parquet` is optional and should be added only when sparse events beyond the current insertion-event fields become useful.

Candidate events:

- `/scoring/insertion_event`
- `/aic/gazebo/contacts/off_limit`

For now, insertion-event fields can remain in `task.parquet` because that path already exists. Off-limit contacts can be promoted later if the topic is consistently present.

## 8. Reward Policy

Do not store a weighted `reward` column.

Store raw components that allow reward recomputation:

- `observation.wrench`
- per-frame `label.*_pose_base`
- `meta/aic/scoring.parquet`
- optional sparse events when available

Training and analysis code should compute reward from these raw components using runtime reward weights. This keeps reward-weight experiments possible without reconverting the dataset.

## 9. Error Handling

The converter should distinguish required and optional data:

- Missing required topics or invalid `validation.json`: skip the run and log a clear reason.
- Missing `episode/metadata.json`: skip the run and log a clear reason.
- Missing optional sparse topics: continue with empty sparse tables or no rows for that episode.
- Missing label pose at a tick: keep the frame with `NaN` pose and `valid=false`.
- Missing entire label source for an episode: keep frames, emit invalid labels, and log a warning.

The conversion summary should include converted count, skipped count, failed count, and the first few skip reasons.

## 10. Testing Requirements

Tests should cover the new behavior without requiring real production MCAP files:

- `validation.json` gating skips invalid or incomplete runs.
- Missing `episode/metadata.json` skips the run.
- `observation.velocity` is reordered by canonical joint order and trimmed with action shift.
- `label.tcp_pose` uses controller state when available.
- TCP fallback can compose a minimal `/tf` chain without ROS runtime dependencies.
- `/scoring/tf` labels are sampled onto 20 Hz ticks and emit `NaN + valid=false` when unavailable.
- `pose_commands.parquet` preserves original command timestamps and does not nearest-fill commands onto frames.
- Existing `meta/aic/{task,scoring,scene,tf_snapshots}.parquet` behavior remains backward compatible.

## 11. Implementation Notes

Implementation should keep extraction boundaries small:

- The existing camera-grid frame extraction should remain focused on cameras, joint state, wrench, and emitted 20 Hz timestamps.
- Auxiliary extractors should run in focused secondary MCAP passes using the emitted 20 Hz timestamps from the frame stream.
- Pose-label extraction should receive the episode metadata, base-frame target names, and emitted timestamps, then return per-frame label arrays plus valid masks.
- Sparse extractors should return rows keyed by `episode_index` and should not mutate the per-frame episode.
- Data conversion helpers should convert message objects into numpy arrays only.
- `DataCreator` should remain responsible for registering LeRobot features and writing frames.
- AIC parquet writers should stay under `src/v3_conversion/aic_meta/`.
