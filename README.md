# rosbag-to-lerobot

ROS2 rosbag (MCAP) files to [LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset converter. No ROS2 installation required.

## Project Structure

```
rosbag-to-lerobot/
├── lerobot/                  # huggingface/lerobot (git submodule)
├── src/
│   ├── config.json           # Conversion configuration
│   ├── main.py               # CLI entry point
│   └── v3_conversion/
│       ├── constants.py      # Path constants (INPUT_PATH, OUTPUT_PATH)
│       ├── converter.py      # Main orchestrator
│       ├── mcap_reader.py    # MCAP frame extraction
│       ├── data_converter.py # Message-to-numpy conversion
│       ├── data_creator.py   # LeRobot dataset writer
│       ├── data_spec.py      # Rosbag config dataclass
│       └── hz_checker.py     # Frequency validation
├── scripts/
│   ├── setup_conda.sh        # Conda environment setup
│   └── run_convert.sh        # Conda conversion runner
├── docker/
│   ├── Dockerfile.conversion
│   └── docker-compose.conversion.yml
└── main.sh                   # Docker launcher
```

## Clone

```bash
git clone https://github.com/Tommoro-AI/rosbag-to-lerobot.git --recursive
cd rosbag-to-lerobot
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Data Layout

Place MCAP files in per-episode folders:

```
<INPUT_DIR>/
├── episode_001/
│   ├── episode_001_0.mcap
│   └── metacard.json         # optional (overrides config.json per episode)
├── episode_002/
│   └── episode_002_0.mcap
└── ...
```

Output is written to `<OUTPUT_DIR>/<task_name>/`.

## Configuration

Edit `src/config.json` to match your data:

```json
{
  "task": "my_task_name",
  "repo_id": "your_hf_org/my_task_name",
  "robot": "ur5e",
  "fps": 30,
  "folders": "all",
  "camera_topic_map": {
    "cam_left": "/left_camera/image",
    "cam_center": "/center_camera/image"
  },
  "joint_names": [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "gripper/left_finger_joint"
  ],
  "state_topic": "/joint_states",
  "action_topics_map": { "leader": "/leader/joint_states" },
  "task_instruction": [],
  "tags": []
}
```

| Field | Description |
|-------|-------------|
| `task` | Task/dataset name (output directory name) |
| `repo_id` | HuggingFace Hub repo (e.g. `org/name`). Set to push after conversion |
| `robot` | Robot type (e.g. `ur5e`) |
| `fps` | Target frames per second |
| `folders` | `"all"` to auto-detect, or list of folder names |
| `camera_topic_map` | Camera name to ROS topic mapping |
| `joint_names` | Ordered joint name list |
| `state_topic` | Observation joint states topic |
| `action_topics_map` | Leader source to ROS topic mapping |

## Run: Docker (main.sh)

**Prerequisites:** Docker & Docker Compose

### Data path configuration

`docker-compose.conversion.yml` mounts two volumes:

| Container path | Host default | Env var |
|----------------|-------------|---------|
| `/data/raw` (read-only) | `./data/raw` | `INPUT_DIR` |
| `/data/lerobot` (read-write) | `./data/lerobot` | `OUTPUT_DIR` |

To use custom paths, set environment variables before running:

```bash
export INPUT_DIR=/path/to/your/mcap/data
export OUTPUT_DIR=/path/to/output
```

### Run

```bash
./main.sh
```

Menu options:
- **1) Cache build & run** — Build Docker image (cached) and start container
- **2) Connect to container** — Attach to running container
- **3) Docker Force Kill** — Stop and remove container
- **5) No-cache rebuild & run** — Full rebuild from scratch

After connecting to the container (option 2):

```bash
python3 main.py                       # uses default config.json
python3 main.py /path/to/config.json  # custom config
```

## Run: Conda

**Prerequisites:** conda (Miniconda or Anaconda), ffmpeg

### 1. Setup environment

```bash
bash scripts/setup_conda.sh              # creates "rosbag2lerobot" env
# or with custom name:
bash scripts/setup_conda.sh my_env_name
```

### 2. Set data paths

Data paths are controlled by environment variables in `src/v3_conversion/constants.py`:

| Env var | Description | Default |
|---------|-------------|---------|
| `INPUT_PATH` | Directory containing episode folders | `/home/weed/aic_results/cheatcode_dataset` |
| `OUTPUT_PATH` | Conversion output directory | `/home/weed/aic_results/lerobot_output` |

Set your paths:

```bash
export INPUT_PATH=/path/to/your/mcap/data
export OUTPUT_PATH=/path/to/output
```

### 3. Run conversion

```bash
conda activate rosbag2lerobot
bash scripts/run_convert.sh
```

Or directly:

```bash
conda activate rosbag2lerobot
export PYTHONPATH=src:lerobot/src
python3 src/main.py
```
