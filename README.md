# rosbag-to-lerobot

Standalone MCAP-to-LeRobot v3.0 dataset conversion pipeline. Converts ROS2 rosbag (MCAP) files into [LeRobot](https://github.com/huggingface/lerobot) v3.0 datasets without requiring a ROS2 installation.

## Prerequisites

- Docker & Docker Compose
- Raw MCAP data organized in per-episode folders

## Project Structure

```
rosbag-to-lerobot/
├── lerobot/                  # huggingface/lerobot (git submodule)
├── src/
│   ├── config.json           # Conversion configuration
│   ├── main.py               # CLI entry point
│   ├── convert.sh            # Container entry script
│   └── v3_conversion/
│       ├── converter.py      # Main orchestrator
│       ├── mcap_reader.py    # MCAP frame extraction
│       ├── data_converter.py # Message-to-numpy conversion
│       ├── data_creator.py   # LeRobot dataset writer
│       ├── data_spec.py      # Rosbag config dataclass
│       ├── hz_checker.py     # Frequency validation
│       └── constants.py      # Path constants
├── docker/
│   ├── Dockerfile.conversion
│   └── docker-compose.conversion.yml
└── main.sh                   # Host-side Docker launcher
```

## Quick Start

### 1. Clone with submodule

```bash
git clone --recurse-submodules https://github.com/Tommoro-AI/rosbag-to-lerobot.git
cd rosbag-to-lerobot
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Prepare raw data

Place MCAP files in per-episode folders:

```
data/raw/
├── episode_001/
│   ├── episode_001_0.mcap    # naming: <folder_name>_0.mcap (or any *.mcap)
│   └── metacard.json         # optional — config.json defaults used if absent
├── episode_002/
│   └── episode_002_0.mcap
└── ...
```

### 3. Edit config.json

Edit `src/config.json` to match your data:

```json
{
  "task": "my_task_name",
  "robot": "ur5e",
  "fps": 30,
  "folders": "all",

  "camera_topic_map": {
    "cam_left": "/left_camera/image",
    "cam_center": "/center_camera/image",
    "cam_right": "/right_camera/image"
  },
  "joint_names": [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper/left_finger_joint"
  ],
  "state_topic": "/joint_states",
  "action_topics_map": {
    "leader": "/leader/joint_states"
  },
  "task_instruction": [],
  "tags": []
}
```

### 4. Run conversion

```bash
./main.sh
# Select option 1 to build & run
# Select option 2 to attach to running container
```

Inside the container:

```bash
python3 main.py                    # uses default config.json
python3 main.py /path/to/config.json  # custom config path
```

Output is written to `/data/lerobot/<task_name>/`.

## Configuration Reference

### config.json fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | yes | Task/dataset name (used as output directory and repo_id) |
| `robot` | string | no | Robot type (e.g., `"ur5e"`) |
| `fps` | int | no | Target frames per second (default: 30) |
| `folders` | list or `"all"` | yes | List of folder names to convert, or `"all"` to auto-detect |
| `camera_topic_map` | object | yes* | Maps camera names to ROS topics |
| `joint_names` | list | yes* | Ordered list of joint names |
| `state_topic` | string | yes* | ROS topic for observation joint states |
| `action_topics_map` | object | yes* | Maps leader sources to ROS topics |
| `task_instruction` | list | no | Task description strings |
| `tags` | list | no | Tags for episode metadata |

> *Required either in `config.json` (as defaults) or in per-folder `metacard.json`.

### metacard.json (optional, per-folder)

When `metacard.json` exists inside an episode folder, its fields **override** the config.json defaults for that episode. This allows mixed configurations across episodes.

### folders: "all"

When set to `"all"`, the pipeline scans `/data/raw/` for all subdirectories containing at least one `*.mcap` file. No `metacard.json` is required.

### Docker volume mapping

The `docker-compose.conversion.yml` mounts:

| Container path | Host default | Env var |
|----------------|-------------|---------|
| `/data/raw` (read-only) | `../data/raw` | `INPUT_DIR` |
| `/data/lerobot` (read-write) | `../data/lerobot` | `OUTPUT_DIR` |

Override with environment variables:

```bash
INPUT_DIR=/path/to/raw OUTPUT_DIR=/path/to/output ./main.sh
```

## Supported Message Types

| Category | ROS2 Message Type |
|----------|-------------------|
| Camera | `sensor_msgs/msg/CompressedImage` |
| Joint state | `sensor_msgs/msg/JointState` |
| Joint trajectory | `trajectory_msgs/msg/JointTrajectory` |
| Odometry | `nav_msgs/msg/Odometry` |
| Twist | `geometry_msgs/msg/Twist` |

## Pipeline Steps

For each episode folder, the converter:

1. Loads metadata from `metacard.json` (or falls back to config defaults)
2. Locates the MCAP file (`<folder>_0.mcap` or first `*.mcap`)
3. Validates that all expected ROS topics exist in the MCAP
4. Extracts synchronized frames at the target FPS
5. Validates actual frequency against the target (min 70% ratio)
6. Converts frames to numpy arrays (observation, action, images)
7. Writes to LeRobot v3.0 dataset format (parquet + video)
8. After all episodes: finalizes dataset, corrects video timestamps, patches metadata

## Troubleshooting

### "camera_topic_map is empty or missing"

Topic configuration was not found. Provide `camera_topic_map` in either `config.json` or the folder's `metacard.json`.

### "No MCAP file found"

The folder exists but contains no `*.mcap` files. Check that MCAP files are in the correct location.

### "MCAP topic pre-check failed"

The configured ROS topics don't exist in the MCAP file. Inspect the MCAP with:

```bash
python3 -c "
from mcap.reader import make_reader
with open('/data/raw/<folder>/<file>.mcap', 'rb') as f:
    for ch in make_reader(f).get_summary().channels.values():
        print(ch.topic)
"
```

### Hz validation failed

The actual message frequency is below the minimum threshold (70% of target FPS). This usually means dropped frames during recording.
