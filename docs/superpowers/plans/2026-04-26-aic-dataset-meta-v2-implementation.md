# AIC Dataset Meta V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the v2 AIC dataset contract: validation-gated conversion, `observation.velocity`, per-frame pose labels with valid masks, and sparse `pose_commands.parquet` while preserving the existing split `meta/aic` parquet layout.

**Architecture:** Keep `extract_frames()` focused on the existing 20 Hz camera-grid stream. Add focused secondary MCAP extractors under `v3_conversion.aic_meta` for pose labels and sparse commands, then wire their outputs into `main.run_conversion()`. Extend `DataCreator` to register/write generic per-frame `label.*` features and `observation.velocity`.

**Tech Stack:** Python 3.10+, pytest, numpy, pyarrow, mcap, mcap-ros2-support, LeRobot v3

**Design reference:** `docs/superpowers/specs/2026-04-26-aic-dataset-and-meta-v2-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `tests/conftest.py` | Extend synthetic MCAP builder with velocity, controller state, `/tf`, and pose commands |
| `src/v3_conversion/aic_meta/sources.py` | Add `load_validation_status()` for `validation.json` gating |
| `src/v3_conversion/data_converter.py` | Extract `/joint_states.velocity` into frame key `velocity` |
| `src/v3_conversion/data_creator.py` | Register/write `observation.velocity` and `label.*` frame features |
| `src/v3_conversion/aic_meta/pose_labels.py` | New secondary extractor for TCP and `/scoring/tf` pose labels |
| `src/v3_conversion/aic_meta/pose_commands.py` | New secondary extractor for `/aic_controller/pose_commands` |
| `src/v3_conversion/aic_meta/schemas.py` | Add `POSE_COMMANDS_SCHEMA` |
| `src/v3_conversion/aic_meta/writer.py` | Add `write_pose_commands_parquet()` |
| `src/main.py` | Apply validation gating, call auxiliary extractors, write new sparse parquet |
| `tests/v3_conversion/test_velocity_feature.py` | Unit tests for velocity extraction and writing |
| `tests/aic_meta/test_validation_status.py` | Unit tests for validation gating source loader |
| `tests/aic_meta/test_pose_labels.py` | Unit tests for pose label extraction |
| `tests/aic_meta/test_pose_commands.py` | Unit tests for sparse command extraction/writing |
| `tests/integration/test_run_conversion.py` | Extend integration coverage for v2 behavior |

## Task 1: Extend Synthetic MCAP Fixture

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_mcap_fixture_aux_topics.py`

- [ ] **Step 1.1: Write fixture behavior tests**

Create `tests/test_mcap_fixture_aux_topics.py`:

```python
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
```

- [ ] **Step 1.2: Run the failing fixture test**

Run:

```bash
pytest tests/test_mcap_fixture_aux_topics.py -q
```

Expected: FAIL because `build_mcap_fixture()` does not accept `controller_state`, `tf`, `pose_commands`, or 4-element `joint_states` tuples.

- [ ] **Step 1.3: Implement fixture support**

Modify `tests/conftest.py`:

1. Add helper message definitions after `_build_image_msgdef()`:

```python
def _build_controller_state_msgdef() -> str:
    definitions = [
        (
            "aic_control_interfaces/msg/ControllerState",
            "std_msgs/Header header\ngeometry_msgs/Pose tcp_pose",
        ),
        ("std_msgs/msg/Header", _load_ros2_msgdef("std_msgs/msg/Header")),
        (
            "builtin_interfaces/msg/Time",
            _load_ros2_msgdef("builtin_interfaces/msg/Time"),
        ),
        ("geometry_msgs/msg/Pose", _load_ros2_msgdef("geometry_msgs/msg/Pose")),
        ("geometry_msgs/msg/Point", _load_ros2_msgdef("geometry_msgs/msg/Point")),
        (
            "geometry_msgs/msg/Quaternion",
            _load_ros2_msgdef("geometry_msgs/msg/Quaternion"),
        ),
    ]
    return "\n================================================\n".join(
        [
            definitions[0][1],
            *[
                f"MSG: {datatype}\n{msgdef}"
                for datatype, msgdef in definitions[1:]
            ],
        ]
    )


def _build_motion_update_msgdef() -> str:
    definitions = [
        (
            "aic_control_interfaces/msg/MotionUpdate",
            "std_msgs/Header header\n"
            "geometry_msgs/Pose pose\n"
            "geometry_msgs/Twist velocity\n"
            "float64[] target_stiffness\n"
            "float64[] target_damping",
        ),
        ("std_msgs/msg/Header", _load_ros2_msgdef("std_msgs/msg/Header")),
        (
            "builtin_interfaces/msg/Time",
            _load_ros2_msgdef("builtin_interfaces/msg/Time"),
        ),
        ("geometry_msgs/msg/Pose", _load_ros2_msgdef("geometry_msgs/msg/Pose")),
        ("geometry_msgs/msg/Point", _load_ros2_msgdef("geometry_msgs/msg/Point")),
        (
            "geometry_msgs/msg/Quaternion",
            _load_ros2_msgdef("geometry_msgs/msg/Quaternion"),
        ),
        ("geometry_msgs/msg/Twist", _load_ros2_msgdef("geometry_msgs/msg/Twist")),
        ("geometry_msgs/msg/Vector3", _load_ros2_msgdef("geometry_msgs/msg/Vector3")),
    ]
    return "\n================================================\n".join(
        [
            definitions[0][1],
            *[
                f"MSG: {datatype}\n{msgdef}"
                for datatype, msgdef in definitions[1:]
            ],
        ]
    )
```

2. Extend `_build()` keyword parameters:

```python
        controller_state=None,
        tf=None,
        pose_commands=None,
```

3. In the joint-state loop, accept optional velocities:

```python
                    for item in messages:
                        if len(item) == 3:
                            t_ns, names, positions = item
                            velocities = [0.0] * len(names)
                        else:
                            t_ns, names, positions, velocities = item
                        queue_message(
                            t_ns,
                            0,
                            topic,
                            joint_state_schema,
                            {
                                "header": {
                                    "stamp": {
                                        "sec": t_ns // 1_000_000_000,
                                        "nanosec": t_ns % 1_000_000_000,
                                    },
                                    "frame_id": "base_link",
                                },
                                "name": list(names),
                                "position": list(positions),
                                "velocity": list(velocities),
                                "effort": [0.0] * len(names),
                            },
                        )
```

4. Add controller-state queueing before image queueing:

```python
            if controller_state:
                controller_schema = writer.register_msgdef(
                    datatype="aic_control_interfaces/msg/ControllerState",
                    msgdef_text=_build_controller_state_msgdef(),
                )
                for t_ns, x, y, z, qx, qy, qz, qw in controller_state:
                    queue_message(
                        t_ns,
                        30,
                        "/aic_controller/controller_state",
                        controller_schema,
                        {
                            "header": {
                                "stamp": {
                                    "sec": t_ns // 1_000_000_000,
                                    "nanosec": t_ns % 1_000_000_000,
                                },
                                "frame_id": "base_link",
                            },
                            "tcp_pose": {
                                "position": {"x": x, "y": y, "z": z},
                                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                            },
                        },
                    )
```

5. Add `/tf` queueing using `_build_tf_message_msgdef()`:

```python
            if tf:
                tf_schema = tf_schema or writer.register_msgdef(
                    datatype="tf2_msgs/msg/TFMessage",
                    msgdef_text=_build_tf_message_msgdef(),
                )
                for t_ns, transforms in tf:
                    tf_msgs = []
                    for parent, child, x, y, z, qx, qy, qz, qw in transforms:
                        tf_msgs.append(
                            {
                                "header": {
                                    "stamp": {
                                        "sec": t_ns // 1_000_000_000,
                                        "nanosec": t_ns % 1_000_000_000,
                                    },
                                    "frame_id": parent,
                                },
                                "child_frame_id": child,
                                "transform": {
                                    "translation": {"x": x, "y": y, "z": z},
                                    "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
                                },
                            }
                        )
                    queue_message(t_ns, 40, "/tf", tf_schema, {"transforms": tf_msgs})
```

6. Add pose-command queueing:

```python
            if pose_commands:
                motion_schema = writer.register_msgdef(
                    datatype="aic_control_interfaces/msg/MotionUpdate",
                    msgdef_text=_build_motion_update_msgdef(),
                )
                for (
                    t_ns,
                    x, y, z, qx, qy, qz, qw,
                    vx, vy, vz, wx, wy, wz,
                    stiffness,
                    damping,
                ) in pose_commands:
                    queue_message(
                        t_ns,
                        50,
                        "/aic_controller/pose_commands",
                        motion_schema,
                        {
                            "header": {
                                "stamp": {
                                    "sec": t_ns // 1_000_000_000,
                                    "nanosec": t_ns % 1_000_000_000,
                                },
                                "frame_id": "base_link",
                            },
                            "pose": {
                                "position": {"x": x, "y": y, "z": z},
                                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                            },
                            "velocity": {
                                "linear": {"x": vx, "y": vy, "z": vz},
                                "angular": {"x": wx, "y": wy, "z": wz},
                            },
                            "target_stiffness": list(stiffness),
                            "target_damping": list(damping),
                        },
                    )
```

- [ ] **Step 1.4: Run fixture tests**

Run:

```bash
pytest tests/test_mcap_fixture_aux_topics.py -q
```

Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add tests/conftest.py tests/test_mcap_fixture_aux_topics.py
git commit -m "test: extend mcap fixture for aic v2 topics"
```

## Task 2: Add `validation.json` Gating Loader

**Files:**
- Modify: `src/v3_conversion/aic_meta/sources.py`
- Create: `tests/aic_meta/test_validation_status.py`

- [ ] **Step 2.1: Write validation loader tests**

Create `tests/aic_meta/test_validation_status.py`:

```python
import json
from pathlib import Path

from v3_conversion.aic_meta.sources import load_validation_status


def test_load_validation_status_accepts_all_passed(tmp_path: Path):
    (tmp_path / "validation.json").write_text(
        json.dumps({"passed_count": 3, "total_count": 3, "checks": []})
    )

    result = load_validation_status(tmp_path)

    assert result["passed"] is True
    assert result["reason"] == ""


def test_load_validation_status_rejects_missing_file(tmp_path: Path):
    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json missing"


def test_load_validation_status_rejects_failed_counts(tmp_path: Path):
    (tmp_path / "validation.json").write_text(
        json.dumps({"passed_count": 2, "total_count": 3, "checks": []})
    )

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json passed_count 2 != total_count 3"


def test_load_validation_status_rejects_failed_check(tmp_path: Path):
    (tmp_path / "validation.json").write_text(
        json.dumps(
            {
                "passed_count": 2,
                "total_count": 2,
                "checks": [
                    {"name": "config.yaml", "passed": True},
                    {"name": "episode/metadata.json", "passed": False},
                ],
            }
        )
    )

    result = load_validation_status(tmp_path)

    assert result["passed"] is False
    assert result["reason"] == "validation.json failed check: episode/metadata.json"
```

- [ ] **Step 2.2: Run the failing tests**

Run:

```bash
pytest tests/aic_meta/test_validation_status.py -q
```

Expected: FAIL because `load_validation_status` does not exist.

- [ ] **Step 2.3: Implement `load_validation_status`**

Append to `src/v3_conversion/aic_meta/sources.py`:

```python
def load_validation_status(run_dir: Path) -> Dict[str, Any]:
    """Return whether validation.json permits conversion for a run."""
    path = run_dir / "validation.json"
    if not path.is_file():
        return {"passed": False, "reason": "validation.json missing"}

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        return {"passed": False, "reason": f"validation.json unreadable: {exc}"}

    checks = raw.get("checks") or []
    for check in checks:
        if isinstance(check, dict) and check.get("passed") is False:
            name = str(check.get("name", "unnamed"))
            return {"passed": False, "reason": f"validation.json failed check: {name}"}

    passed_count = raw.get("passed_count")
    total_count = raw.get("total_count")
    if passed_count is not None and total_count is not None and passed_count != total_count:
        return {
            "passed": False,
            "reason": f"validation.json passed_count {passed_count} != total_count {total_count}",
        }

    return {"passed": True, "reason": ""}
```

- [ ] **Step 2.4: Run validation loader tests**

Run:

```bash
pytest tests/aic_meta/test_validation_status.py -q
```

Expected: PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/v3_conversion/aic_meta/sources.py tests/aic_meta/test_validation_status.py
git commit -m "feat(aic_meta): load validation status"
```

## Task 3: Extract and Write `observation.velocity`

**Files:**
- Modify: `src/v3_conversion/data_converter.py`
- Modify: `src/v3_conversion/data_creator.py`
- Create: `tests/v3_conversion/test_velocity_feature.py`

- [ ] **Step 3.1: Write failing unit tests**

Create `tests/v3_conversion/test_velocity_feature.py`:

```python
from types import SimpleNamespace

import numpy as np

from v3_conversion.data_converter import build_frame, frames_to_episode
from v3_conversion.data_creator import DataCreator


JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper/left_finger_joint",
]


def _joint_msg(names, positions, velocities):
    return SimpleNamespace(
        name=list(names),
        position=list(positions),
        velocity=list(velocities),
        effort=[0.0] * len(names),
    )


def test_build_frame_extracts_velocity_in_canonical_order():
    msg = _joint_msg(
        ["elbow_joint", "shoulder_pan_joint", "shoulder_lift_joint"],
        [0.3, 0.1, 0.2],
        [3.0, 1.0, 2.0],
    )

    frame = build_frame(
        image_msgs={},
        follower_msgs={"observation": msg},
        leader_msgs={"action": msg},
        joint_order={
            "obs": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
            "action": {"action": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"]},
        },
        rot_img=False,
        schema_map={
            "observation": "sensor_msgs/msg/JointState",
            "action": "sensor_msgs/msg/JointState",
        },
    )

    assert np.allclose(frame["obs"], [0.1, 0.2, 0.3])
    assert np.allclose(frame["velocity"], [1.0, 2.0, 3.0])


def test_frames_to_episode_stacks_velocity():
    frames = [
        {
            "images": {},
            "obs": np.array([0.0], dtype=np.float32),
            "velocity": np.array([1.0], dtype=np.float32),
            "action": {"action": np.array([0.1], dtype=np.float32)},
        },
        {
            "images": {},
            "obs": np.array([0.2], dtype=np.float32),
            "velocity": np.array([2.0], dtype=np.float32),
            "action": {"action": np.array([0.3], dtype=np.float32)},
        },
    ]

    episode = frames_to_episode(frames, action_order=["action"], camera_names=[], task="t")

    assert np.allclose(episode["velocity"], [[1.0], [2.0]])


def test_data_creator_registers_and_writes_observation_velocity(tmp_dataset_root):
    creator = DataCreator(
        repo_id="test/velocity",
        root=str(tmp_dataset_root),
        robot_type="ur5e",
        action_order=["action"],
        joint_order={"obs": ["j0"], "action": {"action": ["j0"]}},
        camera_names=[],
        fps=20,
    )
    episode = {
        "obs": np.array([[0.0], [0.2]], dtype=np.float32),
        "velocity": np.array([[1.0], [2.0]], dtype=np.float32),
        "action": np.array([[0.2], [0.4]], dtype=np.float32),
        "images": {},
        "task": "Insert cable.",
    }

    creator.convert_episode(episode)

    assert "observation.velocity" in creator.dataset.features
    assert np.allclose(creator.dataset.frames[0]["observation.velocity"], [1.0])
```

- [ ] **Step 3.2: Run failing tests**

Run:

```bash
pytest tests/v3_conversion/test_velocity_feature.py -q
```

Expected: FAIL because frames do not emit `velocity`, episodes do not stack it, and `DataCreator` does not register it.

- [ ] **Step 3.3: Implement velocity conversion**

Modify `src/v3_conversion/data_converter.py`:

1. Add a velocity helper below `_handle_joint_state`:

```python
def _handle_joint_state_velocity(msg_data, joint_order: List[str]) -> np.ndarray:
    velocity_map = dict(zip(msg_data.name, msg_data.velocity))
    missing = [name for name in joint_order if name not in velocity_map]
    if missing:
        raise KeyError(f"Missing joints in JointState velocity: {missing}")
    ordered = [velocity_map[name] for name in joint_order]
    return np.array(ordered, dtype=np.float32)
```

2. In `build_frame()`, after `follower_data` is built, add:

```python
    velocity_data = None
    obs_msg = (follower_msgs or {}).get("observation")
    obs_schema = schema_map.get("observation", "")
    if obs_msg is not None and obs_schema == "sensor_msgs/msg/JointState":
        velocity_data = _handle_joint_state_velocity(obs_msg, joint_order["obs"])
```

3. Replace the result construction with:

```python
    result = {"images": camera_data, "obs": follower_data, "action": action_data}
    if velocity_data is not None:
        result["velocity"] = velocity_data
```

4. In `frames_to_episode()`, add `velocity_list = []`, append `f["velocity"]` when present, validate all-or-none like wrench, and set:

```python
    if velocity_list:
        episode["velocity"] = np.stack(velocity_list, axis=0)
```

- [ ] **Step 3.4: Implement velocity writing**

Modify `src/v3_conversion/data_creator.py`:

1. In `create_dataset()`, after `observation.state`, add:

```python
        if "velocity" in episode:
            features["observation.velocity"] = {
                "dtype": "float32",
                "shape": (obs_dim,),
                "names": self.joint_order["obs"],
            }
```

2. In `convert_episode()`, after `obs = ...`, add:

```python
        velocity = (
            np.asarray(episode["velocity"], dtype=np.float32)
            if "velocity" in episode
            else None
        )
```

3. Validate feature consistency near wrench validation:

```python
        dataset_has_velocity = "observation.velocity" in self.dataset.features
        episode_has_velocity = velocity is not None
        if dataset_has_velocity and not episode_has_velocity:
            raise ValueError(
                "Dataset expects observation.velocity but this episode does not provide velocity data."
            )
        if episode_has_velocity and not dataset_has_velocity:
            raise ValueError(
                "This episode provides velocity data but the dataset was created without observation.velocity."
            )
        if episode_has_velocity and len(velocity) != frame_count:
            raise ValueError(
                f"Velocity has {len(velocity)} frames, expected {frame_count}"
            )
```

4. In the per-frame loop, add:

```python
            if velocity is not None:
                frame["observation.velocity"] = velocity[t]
```

- [ ] **Step 3.5: Run velocity tests**

Run:

```bash
pytest tests/v3_conversion/test_velocity_feature.py tests/v3_conversion/test_action_shift.py -q
```

Expected: PASS. `test_action_shift.py` confirms extra per-frame numpy arrays trim to `T - 1`.

- [ ] **Step 3.6: Commit**

```bash
git add src/v3_conversion/data_converter.py src/v3_conversion/data_creator.py tests/v3_conversion/test_velocity_feature.py
git commit -m "feat: add observation velocity feature"
```

## Task 4: Add Generic `label.*` Feature Support

**Files:**
- Modify: `src/v3_conversion/data_creator.py`
- Create: `tests/v3_conversion/test_data_creator_labels.py`

- [ ] **Step 4.1: Write failing label feature tests**

Create `tests/v3_conversion/test_data_creator_labels.py`:

```python
import numpy as np

from v3_conversion.data_creator import DataCreator


def test_data_creator_registers_and_writes_label_features(tmp_dataset_root):
    creator = DataCreator(
        repo_id="test/labels",
        root=str(tmp_dataset_root),
        robot_type="ur5e",
        action_order=["action"],
        joint_order={"obs": ["j0"], "action": {"action": ["j0"]}},
        camera_names=[],
        fps=20,
    )
    episode = {
        "obs": np.array([[0.0], [0.2]], dtype=np.float32),
        "action": np.array([[0.2], [0.4]], dtype=np.float32),
        "label.tcp_pose": np.array(
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "label.tcp_pose_valid": np.array([True, False], dtype=np.bool_),
        "images": {},
        "task": "Insert cable.",
    }

    creator.convert_episode(episode)

    assert creator.dataset.features["label.tcp_pose"]["shape"] == (7,)
    assert creator.dataset.features["label.tcp_pose_valid"]["dtype"] == "bool"
    assert np.allclose(creator.dataset.frames[0]["label.tcp_pose"], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert creator.dataset.frames[0]["label.tcp_pose_valid"] is True
```

- [ ] **Step 4.2: Run failing label tests**

Run:

```bash
pytest tests/v3_conversion/test_data_creator_labels.py -q
```

Expected: FAIL because `DataCreator` ignores `label.*` episode keys.

- [ ] **Step 4.3: Implement label feature registration and writing**

Modify `src/v3_conversion/data_creator.py`:

1. In `create_dataset()`, before camera feature registration, add:

```python
        for key in sorted(k for k in episode.keys() if k.startswith("label.")):
            arr = np.asarray(episode[key])
            feature_dtype = "bool" if arr.dtype == np.bool_ else "float32"
            feature_shape = tuple(arr.shape[1:])
            features[key] = {
                "dtype": feature_dtype,
                "shape": feature_shape,
                "names": None,
            }
```

2. In `convert_episode()`, after camera length validation, add:

```python
        label_arrays: Dict[str, np.ndarray] = {}
        for key in sorted(k for k in episode.keys() if k.startswith("label.")):
            arr = np.asarray(episode[key])
            if len(arr) != frame_count:
                raise ValueError(f"{key} has {len(arr)} frames, expected {frame_count}")
            if key not in self.dataset.features:
                raise ValueError(
                    f"This episode provides {key} but the dataset was created without that feature."
                )
            label_arrays[key] = arr
```

3. In the per-frame loop, before `self.dataset.add_frame(frame)`, add:

```python
            for key, arr in label_arrays.items():
                value = arr[t]
                if arr.dtype == np.bool_ and value.shape == ():
                    frame[key] = bool(value)
                else:
                    frame[key] = value.astype(np.float32, copy=False)
```

- [ ] **Step 4.4: Run label tests**

Run:

```bash
pytest tests/v3_conversion/test_data_creator_labels.py tests/v3_conversion/test_action_shift.py -q
```

Expected: PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/v3_conversion/data_creator.py tests/v3_conversion/test_data_creator_labels.py
git commit -m "feat(data_creator): write label frame features"
```

## Task 5: Implement Pose Label Extraction

**Files:**
- Create: `src/v3_conversion/aic_meta/pose_labels.py`
- Create: `tests/aic_meta/test_pose_labels.py`

- [ ] **Step 5.1: Write failing pose-label tests**

Create `tests/aic_meta/test_pose_labels.py`:

```python
import math
from pathlib import Path

import numpy as np
import pytest

from v3_conversion.aic_meta.pose_labels import (
    extract_pose_labels,
    frame_id_candidates,
)


def test_frame_id_candidates_cover_real_aic_suffixes():
    candidates = frame_id_candidates(
        name="sfp_tip",
        cable_name="cable_0",
        target_module="nic_card_mount_0",
    )

    assert "sfp_tip" in candidates
    assert "sfp_tip_link" in candidates
    assert "cable_0/sfp_tip_link" in candidates


def test_extract_pose_labels_uses_controller_state_for_tcp(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "tcp_controller.mcap",
        controller_state=[
            (0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            (50_000_000, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )

    labels = extract_pose_labels(
        bag_path=bag,
        frame_timestamps_ns=[0, 50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert np.allclose(labels["label.tcp_pose"][0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert labels["label.tcp_pose_valid"].tolist() == [True, True]


def test_extract_pose_labels_uses_tf_fallback_for_tcp(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "tcp_tf.mcap",
        tf=[
            (0, [("base_link", "tcp_link", 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)]),
        ],
    )

    labels = extract_pose_labels(
        bag_path=bag,
        frame_timestamps_ns=[0],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert np.allclose(labels["label.tcp_pose"][0], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert labels["label.tcp_pose_valid"].tolist() == [True]


def test_extract_pose_labels_samples_scoring_tf_and_masks_missing(
    build_mcap_fixture, tmp_path: Path
):
    bag = build_mcap_fixture(
        path=tmp_path / "scoring_labels.mcap",
        scoring_tf=[
            (
                50_000_000,
                [
                    ("base_link", "cable_0/sfp_tip_link", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                    ("base_link", "task_board/nic_card_mount_0/sfp_port_0_link", 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                    ("base_link", "task_board/nic_card_mount_0", 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                ],
            )
        ],
    )

    labels = extract_pose_labels(
        bag_path=bag,
        frame_timestamps_ns=[0, 50_000_000],
        episode_meta={
            "cable_name": "cable_0",
            "plug_name": "sfp_tip",
            "port_name": "sfp_port_0",
            "target_module": "nic_card_mount_0",
        },
        base_frame="base_link",
    )

    assert labels["label.plug_pose_base_valid"].tolist() == [False, True]
    assert math.isnan(float(labels["label.plug_pose_base"][0][0]))
    assert np.allclose(labels["label.plug_pose_base"][1], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(labels["label.port_pose_base"][1], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(labels["label.target_module_pose_base"][1], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
```

- [ ] **Step 5.2: Run failing pose-label tests**

Run:

```bash
pytest tests/aic_meta/test_pose_labels.py -q
```

Expected: FAIL because `pose_labels.py` does not exist.

- [ ] **Step 5.3: Implement `pose_labels.py`**

Create `src/v3_conversion/aic_meta/pose_labels.py` with these public functions and constants:

```python
"""Per-frame AIC pose label extraction from MCAP auxiliary topics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory


POSE_LABEL_KEYS = (
    "label.tcp_pose",
    "label.plug_pose_base",
    "label.port_pose_base",
    "label.target_module_pose_base",
)


def _nan_pose(count: int) -> np.ndarray:
    return np.full((count, 7), np.nan, dtype=np.float32)


def _pose_from_pose_msg(pose: Any) -> np.ndarray:
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


def _pose_from_transform(transform: Any) -> np.ndarray:
    return np.array(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ],
        dtype=np.float32,
    )


def frame_id_candidates(name: str, cable_name: str = "", target_module: str = "") -> List[str]:
    """Return likely TF child-frame IDs for an AIC semantic object name."""
    base = str(name or "").strip()
    candidates: list[str] = []
    if base:
        candidates.extend([base, f"{base}_link", f"{base}_link_entrance"])
    if cable_name and base:
        candidates.extend(
            [
                f"{cable_name}/{base}",
                f"{cable_name}/{base}_link",
                f"{cable_name}/{base}_link_entrance",
            ]
        )
    if target_module and base:
        candidates.extend(
            [
                f"task_board/{target_module}/{base}",
                f"task_board/{target_module}/{base}_link",
                f"task_board/{target_module}/{base}_link_entrance",
            ]
        )
    if target_module and base == target_module:
        candidates.extend([f"task_board/{target_module}", f"task_board/{target_module}/{target_module}_link"])
    return list(dict.fromkeys(candidates))
```

Also implement:

```python
def _decoded_messages(bag_path: Path, wanted_topics: set[str]):
    decoder_factory = DecoderFactory()
    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    decoders: dict[int, Any] = {}
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
                    if channel is None or channel.topic not in wanted_topics:
                        continue
                    schema = schemas.get(channel.schema_id)
                    if schema is None:
                        continue
                    if record.channel_id not in decoders:
                        decoders[record.channel_id] = decoder_factory.decoder_for(
                            channel.message_encoding, schema
                        )
                    decoder = decoders[record.channel_id]
                    if decoder is None:
                        continue
                    try:
                        yield channel.topic, int(record.log_time), decoder(record.data)
                    except Exception:
                        continue
    except Exception:
        return
```

Then implement sampling helpers:

```python
def _sample_previous(samples: list[tuple[int, np.ndarray]], t_ns: int) -> tuple[np.ndarray | None, bool]:
    selected = None
    for sample_t, pose in samples:
        if sample_t <= t_ns:
            selected = pose
        else:
            break
    if selected is None:
        return None, False
    return selected, True


def _fill_from_samples(target: np.ndarray, valid: np.ndarray, samples: list[tuple[int, np.ndarray]], frame_timestamps_ns: list[int]) -> None:
    samples.sort(key=lambda item: item[0])
    for idx, t_ns in enumerate(frame_timestamps_ns):
        pose, ok = _sample_previous(samples, t_ns)
        if ok:
            target[idx] = pose
            valid[idx] = True
```

And the public extractor:

```python
def extract_pose_labels(
    bag_path: Path,
    frame_timestamps_ns: list[int],
    episode_meta: Dict[str, Any],
    base_frame: str = "base_link",
) -> Dict[str, np.ndarray]:
    count = len(frame_timestamps_ns)
    result: Dict[str, np.ndarray] = {
        key: _nan_pose(count) for key in POSE_LABEL_KEYS
    }
    result.update({f"{key}_valid": np.zeros((count,), dtype=np.bool_) for key in POSE_LABEL_KEYS})

    tcp_samples: list[tuple[int, np.ndarray]] = []
    tf_tcp_samples: list[tuple[int, np.ndarray]] = []
    scoring_samples: dict[str, list[tuple[int, np.ndarray]]] = {
        "label.plug_pose_base": [],
        "label.port_pose_base": [],
        "label.target_module_pose_base": [],
    }

    plug_candidates = set(frame_id_candidates(
        str(episode_meta.get("plug_name", "")),
        cable_name=str(episode_meta.get("cable_name", "")),
        target_module=str(episode_meta.get("target_module", "")),
    ))
    port_candidates = set(frame_id_candidates(
        str(episode_meta.get("port_name", "")),
        cable_name=str(episode_meta.get("cable_name", "")),
        target_module=str(episode_meta.get("target_module", "")),
    ))
    target_candidates = set(frame_id_candidates(
        str(episode_meta.get("target_module", "")),
        cable_name=str(episode_meta.get("cable_name", "")),
        target_module=str(episode_meta.get("target_module", "")),
    ))

    for topic, t_ns, msg in _decoded_messages(
        Path(bag_path),
        {"/aic_controller/controller_state", "/tf", "/scoring/tf"},
    ):
        if topic == "/aic_controller/controller_state" and hasattr(msg, "tcp_pose"):
            tcp_samples.append((t_ns, _pose_from_pose_msg(msg.tcp_pose)))
            continue

        transforms = getattr(msg, "transforms", [])
        if topic == "/tf":
            for tf in transforms:
                if tf.header.frame_id == base_frame and tf.child_frame_id in {"tcp_link", "tool0", "tool_link"}:
                    tf_tcp_samples.append((t_ns, _pose_from_transform(tf.transform)))
            continue

        if topic == "/scoring/tf":
            for tf in transforms:
                child = tf.child_frame_id
                pose = _pose_from_transform(tf.transform)
                if child in plug_candidates:
                    scoring_samples["label.plug_pose_base"].append((t_ns, pose))
                if child in port_candidates:
                    scoring_samples["label.port_pose_base"].append((t_ns, pose))
                if child in target_candidates:
                    scoring_samples["label.target_module_pose_base"].append((t_ns, pose))

    tcp_source = tcp_samples if tcp_samples else tf_tcp_samples
    _fill_from_samples(
        result["label.tcp_pose"],
        result["label.tcp_pose_valid"],
        tcp_source,
        frame_timestamps_ns,
    )
    for key, samples in scoring_samples.items():
        _fill_from_samples(
            result[key],
            result[f"{key}_valid"],
            samples,
            frame_timestamps_ns,
        )
    return result
```

- [ ] **Step 5.4: Run pose-label tests**

Run:

```bash
pytest tests/aic_meta/test_pose_labels.py -q
```

Expected: PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/v3_conversion/aic_meta/pose_labels.py tests/aic_meta/test_pose_labels.py
git commit -m "feat(aic_meta): extract per-frame pose labels"
```

## Task 6: Add Sparse Pose Command Parquet

**Files:**
- Create: `src/v3_conversion/aic_meta/pose_commands.py`
- Modify: `src/v3_conversion/aic_meta/schemas.py`
- Modify: `src/v3_conversion/aic_meta/writer.py`
- Create: `tests/aic_meta/test_pose_commands.py`

- [ ] **Step 6.1: Write failing pose-command tests**

Create `tests/aic_meta/test_pose_commands.py`:

```python
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from v3_conversion.aic_meta.pose_commands import extract_pose_commands
from v3_conversion.aic_meta.writer import write_pose_commands_parquet


def test_extract_pose_commands_preserves_sparse_times(build_mcap_fixture, tmp_path: Path):
    bag = build_mcap_fixture(
        path=tmp_path / "commands.mcap",
        pose_commands=[
            (
                1_000_000_000,
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                [90.0] * 36,
                [20.0] * 36,
            ),
            (
                1_500_000_000,
                4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                [91.0] * 36,
                [21.0] * 36,
            ),
        ],
    )

    rows = extract_pose_commands(
        bag_path=bag,
        episode_index=7,
        episode_start_ns=1_000_000_000,
    )

    assert [row["episode_index"] for row in rows] == [7, 7]
    assert [row["t_ns"] for row in rows] == [1_000_000_000, 1_500_000_000]
    assert rows[0]["time_sec"] == pytest.approx(0.0)
    assert rows[1]["time_sec"] == pytest.approx(0.5)
    assert rows[0]["pose"] == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    assert rows[0]["velocity"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_write_pose_commands_parquet_roundtrip(tmp_path: Path):
    target = tmp_path / "aic" / "pose_commands.parquet"
    rows = [
        {
            "episode_index": 7,
            "t_ns": 1_000_000_000,
            "time_sec": 0.0,
            "pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            "velocity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "stiffness": [90.0] * 36,
            "damping": [20.0] * 36,
        }
    ]

    write_pose_commands_parquet(target, rows)

    loaded = pq.read_table(target).to_pylist()
    assert loaded[0]["episode_index"] == 7
    assert loaded[0]["pose"][0] == pytest.approx(1.0)
    assert len(loaded[0]["stiffness"]) == 36
```

- [ ] **Step 6.2: Run failing pose-command tests**

Run:

```bash
pytest tests/aic_meta/test_pose_commands.py -q
```

Expected: FAIL because `pose_commands.py`, schema, and writer do not exist.

- [ ] **Step 6.3: Implement `pose_commands.py`**

Create `src/v3_conversion/aic_meta/pose_commands.py`:

```python
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
                    if channel is None or channel.topic != "/aic_controller/pose_commands":
                        continue
                    schema = schemas.get(channel.schema_id)
                    if schema is None:
                        continue
                    if record.channel_id not in decoders:
                        decoders[record.channel_id] = decoder_factory.decoder_for(
                            channel.message_encoding, schema
                        )
                    decoder = decoders[record.channel_id]
                    if decoder is None:
                        continue
                    try:
                        msg = decoder(record.data)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "episode_index": int(episode_index),
                            "t_ns": int(record.log_time),
                            "time_sec": float((int(record.log_time) - episode_start_ns) / 1_000_000_000),
                            "pose": _pose_list(msg.pose),
                            "velocity": _twist_list(msg.velocity),
                            "stiffness": [float(x) for x in getattr(msg, "target_stiffness", [])],
                            "damping": [float(x) for x in getattr(msg, "target_damping", [])],
                        }
                    )
    except Exception:
        return rows
    return rows
```

- [ ] **Step 6.4: Add schema and writer**

Modify `src/v3_conversion/aic_meta/schemas.py`:

```python
POSE_COMMANDS_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("t_ns", pa.int64(), nullable=False),
    pa.field("time_sec", pa.float32()),
    pa.field("pose", pa.list_(pa.float32(), list_size=7)),
    pa.field("velocity", pa.list_(pa.float32(), list_size=6)),
    pa.field("stiffness", pa.list_(pa.float32())),
    pa.field("damping", pa.list_(pa.float32())),
])
```

Modify `src/v3_conversion/aic_meta/writer.py`:

```python
def write_pose_commands_parquet(target: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _write_rows(target, rows, schemas.POSE_COMMANDS_SCHEMA)
```

- [ ] **Step 6.5: Run pose-command tests**

Run:

```bash
pytest tests/aic_meta/test_pose_commands.py tests/aic_meta/test_schemas.py tests/aic_meta/test_writer.py -q
```

Expected: PASS.

- [ ] **Step 6.6: Commit**

```bash
git add src/v3_conversion/aic_meta/pose_commands.py src/v3_conversion/aic_meta/schemas.py src/v3_conversion/aic_meta/writer.py tests/aic_meta/test_pose_commands.py
git commit -m "feat(aic_meta): write sparse pose commands"
```

## Task 7: Wire V2 Data Into `main.run_conversion`

**Files:**
- Modify: `src/main.py`
- Modify: `tests/test_main_conversion.py`
- Modify: `tests/integration/test_run_conversion.py`

- [ ] **Step 7.1: Add main-level tests for validation skip**

In `tests/test_main_conversion.py`, add a test near the existing failure-path tests:

```python
def test_run_conversion_skips_when_validation_fails(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    run_dir = input_dir / "run_bad"
    run_dir.mkdir(parents=True)
    (run_dir / "validation.json").write_text('{"passed_count": 0, "total_count": 1, "checks": []}')
    config_path.write_text(
        json.dumps(
            {
                "task": "aic_task",
                "repo_id": "local/aic_task",
                "robot": "ur5e",
                "fps": 20,
                "folders": ["run_bad"],
                "camera_topic_map": {"cam_left": "/left_camera/image"},
                "joint_names": ["j0"],
                "state_topic": "/joint_states",
                "action_topics_map": {"leader": "/joint_states"},
            }
        )
    )

    exit_code = main.run_conversion(str(config_path), input_dir=str(input_dir), output_dir=str(output_dir))

    assert exit_code == 1
    assert "Skipping run_bad" in capsys.readouterr().err
```

- [ ] **Step 7.2: Run failing main skip test**

Run:

```bash
pytest tests/test_main_conversion.py::test_run_conversion_skips_when_validation_fails -q
```

Expected: FAIL because `run_conversion()` does not load validation status or log a skip before preparing config.

- [ ] **Step 7.3: Wire validation gating and auxiliary extractors**

Modify `src/main.py` imports:

```python
from v3_conversion.aic_meta.pose_commands import extract_pose_commands
from v3_conversion.aic_meta.pose_labels import extract_pose_labels
from v3_conversion.aic_meta.sources import (
    load_episode_metadata,
    load_run_meta,
    load_scene_from_config,
    load_scoring_yaml,
    load_tags,
    load_validation_status,
)
from v3_conversion.aic_meta.writer import (
    write_pose_commands_parquet,
    write_scene_parquet,
    write_scoring_parquet,
    write_task_parquet,
    write_tf_snapshots_parquet,
)
```

Add row storage near existing AIC row lists:

```python
        aic_pose_command_rows = _load_existing_parquet_rows(aic_dir / "pose_commands.parquet")
```

and in non-merge mode:

```python
        aic_pose_command_rows = []
```

Before the per-folder loop, add skip accounting:

```python
    skipped_count = 0
    skipped_reasons: List[str] = []
```

At the start of each per-folder iteration, before loading the metacard, apply validation as a skip, not a failure:

```python
        run_dir = INPUT_PATH / folder_name
        validation_status = load_validation_status(run_dir)
        if not validation_status["passed"]:
            skipped_count += 1
            reason = f"Skipping {folder_name}: {validation_status['reason']}"
            skipped_reasons.append(reason)
            logger.warning("  %s", reason)
            continue
        has_root_episode_meta = (run_dir / "episode" / "metadata.json").is_file()
        has_trial_episode_meta = any(
            (trial_dir / "episode" / "metadata.json").is_file()
            for trial_dir in run_dir.glob("trial_*")
            if trial_dir.is_dir()
        )
        if not has_root_episode_meta and not has_trial_episode_meta:
            skipped_count += 1
            reason = f"Skipping {folder_name}: episode/metadata.json missing"
            skipped_reasons.append(reason)
            logger.warning("  %s", reason)
            continue

        try:
            # existing per-folder conversion body continues here
```

Keep the existing trial-layout detection, but remove the later duplicate `run_dir = INPUT_PATH / folder_name`.

Before `frames_to_episode()` consumes frames, capture timestamps and extract labels:

```python
            frame_timestamps_ns = [
                int(frame["emitted_timestamp_ns"]) for frame in frames
            ]
            pose_labels = extract_pose_labels(
                bag_path=mcap_path,
                frame_timestamps_ns=frame_timestamps_ns,
                episode_meta=episode_meta,
                base_frame="base_link",
            )
```

After `episode = frames_to_episode(...)`, attach labels before shifting:

```python
            episode.update(pose_labels)
```

After `episode_start_ns` is known, collect sparse commands:

```python
            pose_command_rows = extract_pose_commands(
                bag_path=mcap_path,
                episode_index=ep_idx,
                episode_start_ns=episode_start_ns,
            )
```

After successful `creator.convert_episode(episode)`, append:

```python
            aic_pose_command_rows.extend(pose_command_rows)
```

During finalization, write:

```python
            write_pose_commands_parquet(
                aic_dir / "pose_commands.parquet",
                aic_pose_command_rows,
            )
```

In the summary block, include skip information:

```python
    logger.info(
        "Conversion summary: converted=%d failed=%d skipped=%d total=%d",
        converted_count,
        failed_count,
        skipped_count,
        len(folders),
    )
    for reason in skipped_reasons[:10]:
        logger.warning("  %s", reason)
```

- [ ] **Step 7.4: Extend integration test expectations**

In `tests/integration/test_run_conversion.py`, extend the fixture run to include:

```python
controller_state = [
    (i * 20_000_000, 1.0 + i, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    for i in range(7)
]
pose_commands = [
    (
        40_000_000,
        1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        [90.0] * 36,
        [20.0] * 36,
    )
]
```

Pass those into `build_mcap_fixture(...)`. Then assert:

```python
    assert (aic_dir / "pose_commands.parquet").is_file()
    pose_command_rows = pq.read_table(aic_dir / "pose_commands.parquet").to_pylist()
    assert len(pose_command_rows) == 1
    assert pose_command_rows[0]["time_sec"] >= 0.0

    data_table = pq.read_table(next((output_root / "data").rglob("*.parquet")))
    assert "observation.velocity" in data_table.column_names
    assert "label.tcp_pose" in data_table.column_names
    assert "label.tcp_pose_valid" in data_table.column_names
```

- [ ] **Step 7.5: Run focused main and integration tests**

Run:

```bash
pytest tests/test_main_conversion.py::test_run_conversion_skips_when_validation_fails tests/integration/test_run_conversion.py -q
```

Expected: PASS.

- [ ] **Step 7.6: Commit**

```bash
git add src/main.py tests/test_main_conversion.py tests/integration/test_run_conversion.py
git commit -m "feat(main): wire aic v2 labels and sparse commands"
```

## Task 8: Backward Compatibility and Full Verification

**Files:**
- Modify only files needed to fix regressions found by the commands below.

- [ ] **Step 8.1: Run AIC metadata tests**

Run:

```bash
pytest tests/aic_meta -q
```

Expected: PASS.

- [ ] **Step 8.2: Run v3 conversion unit tests**

Run:

```bash
pytest tests/v3_conversion -q
```

Expected: PASS.

- [ ] **Step 8.3: Run main and integration tests**

Run:

```bash
pytest tests/test_main_conversion.py tests/integration/test_run_conversion.py -q
```

Expected: PASS.

- [ ] **Step 8.4: Run full local test suite**

Run:

```bash
pytest -q
```

Expected: PASS. If tests requiring local LeRobot dependencies fail because the dependency is absent, record the exact missing module or command output in the final handoff and run the narrower passing suites from Steps 8.1 through 8.3.

- [ ] **Step 8.5: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional source/test changes remain staged or unstaged. Existing unrelated dirty files from before this plan should not be reverted.

- [ ] **Step 8.6: Commit final verification fixes**

If Step 8 required fixes, commit them:

```bash
git add src tests
git commit -m "test: verify aic dataset meta v2 conversion"
```

If Step 8 required no fixes, do not create an empty commit.

## Spec Coverage Checklist

| Spec requirement | Plan coverage |
|---|---|
| Validation gating | Task 2, Task 7 |
| Skip missing `episode/metadata.json` | Task 7 |
| `observation.velocity` | Task 3 |
| Per-frame `label.*` features and valid masks | Task 4, Task 5, Task 7 |
| TCP controller-state primary source | Task 5 |
| `/tf` fallback for TCP | Task 5 |
| `/scoring/tf` 20 Hz pose labels, no raw storage | Task 5, Task 7 |
| Split `meta/aic` retained | Task 6, Task 7, Task 8 |
| `pose_commands.parquet` sparse table | Task 6, Task 7 |
| No precomputed reward or wrench norm | Task 7 does not add these columns |
| Backward compatibility | Task 8 |
