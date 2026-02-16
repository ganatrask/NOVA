<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-Physical%20AI-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA Physical AI"/>
  <img src="https://img.shields.io/badge/GR00T-N1.6-00D4AA?style=for-the-badge" alt="GR00T N1.6"/>
  <img src="https://img.shields.io/badge/Cosmos-Reason%202-FF6B00?style=for-the-badge" alt="Cosmos Reason 2"/>
  <img src="https://img.shields.io/badge/Parakeet-ASR-0066CC?style=for-the-badge" alt="Parakeet ASR"/>
</p>

<h1 align="center">NOVA</h1>
<h3 align="center">Neural Open Vision Actions</h3>

<p align="center">
  <strong>End-to-end Physical AI pipeline combining voice commands, scene reasoning, and learned manipulation</strong>
</p>

<p align="center">
  <a href="https://github.com/ganatrask/NOVA">
    <img src="https://img.shields.io/badge/GitHub-NOVA-181717?style=flat-square&logo=github" alt="GitHub"/>
  </a>
  <a href="https://huggingface.co/datasets/ganatrask/NOVA">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?style=flat-square&logo=huggingface" alt="Dataset"/>
  </a>
  <a href="https://huggingface.co/ganatrask/NOVA">
    <img src="https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=flat-square&logo=huggingface" alt="Model"/>
  </a>
  <img src="https://img.shields.io/badge/Episodes-100-blue?style=flat-square" alt="Episodes"/>
  <img src="https://img.shields.io/badge/Training-30K%20steps-green?style=flat-square" alt="Training"/>
</p>

<p align="center">
  Built with <b>NVIDIA Cosmos Reason 2</b> + <b>GR00T N1.6</b> + <b>Parakeet ASR</b>
</p>

---

## What is NOVA?

NOVA demonstrates the **complete NVIDIA Physical AI stack** working together on [Pollen Robotics' Reachy 2](https://www.pollen-robotics.com/reachy/) humanoid robot:

```
   "Pick up the red cube"
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         NOVA PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   VOICE INPUT                    SCENE UNDERSTANDING                │
│   ┌─────────────┐               ┌──────────────────┐               │
│   │  Parakeet   │──────────────▶│  Cosmos Reason 2 │               │
│   │    ASR      │   text        │  "I see a red    │               │
│   │  (600M)     │   command     │   cube at..."    │               │
│   └─────────────┘               └────────┬─────────┘               │
│                                          │                         │
│                                          ▼                         │
│                              ┌──────────────────┐                  │
│                              │    GR00T N1.6    │                  │
│                              │   Action Policy  │                  │
│                              │  (Fine-tuned)    │                  │
│                              └────────┬─────────┘                  │
│                                       │                            │
│                                       ▼                            │
│                              ┌──────────────────┐                  │
│                              │    Reachy 2      │                  │
│                              │  (14-DOF Arms)   │                  │
│                              └──────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

| Component | Technology | What It Does |
|-----------|------------|--------------|
| **Voice Input** | Parakeet CTC 0.6B | 6% WER, 50x real-time transcription |
| **Scene Reasoning** | Cosmos Reason 2 | Object detection, spatial understanding, task planning |
| **Action Policy** | GR00T N1.6 | Vision-language-action model for manipulation |
| **Simulation** | MuJoCo + reachy2_mujoco | High-fidelity physics with domain randomization |
| **Dataset** | LeRobot v2.1 | 100 expert  episodes, HuggingFace compatible |

## Results

### Dataset Collection
- **100 episodes** of pick-and-place demonstrations
- **32 task variations** (4 objects × 8 colors)
- **Domain randomization**: position, lighting, camera jitter
- **Format**: LeRobot v2.1 with parquet + H264 video

### GR00T Training
| Metric | Value |
|--------|-------|
| GPU | NVIDIA A100-SXM4-80GB |
| Training Steps | 30,000 |
| Batch Size | 64 |
| Final Loss | ~0.008-0.01 |
| Checkpoints | 15K, 21K, 24K, 30K |

### Objects & Colors
<table>
<tr>
<td>

**Objects**
- Cube (4cm)
- Rectangular box
- Cylinder
- Capsule

</td>
<td>

**Colors**
- Red, Green, Blue, Yellow
- Cyan, Magenta, Orange, Purple

</td>
</tr>
</table>

## Quick Start

### Prerequisites
```bash
# Create conda environments
conda create -n reachy_groot python=3.10 -y
conda create -n reachy_cosmos python=3.10 -y

# Clone repository with submodules
git clone --recurse-submodules https://github.com/ganatrask/NOVA.git
cd NOVA

# If already cloned without submodules:
git submodule update --init --recursive
```

### Installation
```bash
# Activate main environment
conda activate reachy_groot

# Install reachy2_mujoco (simulation)
pip install -e libs/reachy2_mujoco

# Install Isaac-GR00T (action policy)
cd Isaac-GR00T
pip install uv && uv sync --python 3.10 && uv pip install -e .
cd ..

# Install remaining dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

**Terminal 1: MuJoCo Server**
```bash
reachy2-mujoco --headless
```

**Terminal 2: Cosmos Reasoning Server**
```bash
conda activate reachy_cosmos
python scripts/cosmos_server.py --port 8100
```

**Terminal 3: Gradio GUI**
```bash
conda activate reachy_groot
python scripts/pipeline_gui.py --model-path checkpoints/groot-reachy2-pickplace/checkpoint-30000
```

Open http://localhost:7860 in your browser.

## Architecture Deep Dive

### Data Collection Pipeline
```
MuJoCo Simulation
       │
       ├── Domain Randomization
       │   ├── Object: cube, rect_box, cylinder, capsule
       │   ├── Color: 8 variations
       │   ├── Position: workspace-aware random
       │   ├── Lighting: 0.5-1.0 intensity
       │   └── Camera: ±2° jitter
       │
       ├── External Cameras
       │   ├── front_cam (640×480, 108° FOV)
       │   └── workspace_cam (640×480, 70° FOV)
       │
       └── LeRobot v2.1 Dataset
           ├── Parquet files (states + actions)
           ├── MP4 videos (H264)
           └── stats.json (normalization)
```

### GR00T Action Space
```python
# Right arm (8 values)
action = [
    shoulder_pitch,  # -180° to 90°
    shoulder_roll,   # -180° to 10°
    elbow_yaw,       # -90° to 90°
    elbow_pitch,     # -125° to 0°
    wrist_roll,      # -100° to 100°
    wrist_pitch,     # -45° to 45°
    wrist_yaw,       # -30° to 30°
    gripper,         # 0 (closed) to 1 (open)
]
```

### Cosmos Reason 2 Integration
```python
# Scene analysis output
{
    "objects": [
        {"object": "cube", "color": "red", "position": "center-right"},
        {"object": "box", "color": "white", "position": "center"}
    ],
    "gr00t_instruction": "Pick up the red cube and place it in the white box"
}
```

## Project Structure

```
NOVA/
├── scripts/
│   ├── pipeline_gui.py          # Gradio web interface
│   ├── data_collector.py        # Dataset collection
│   ├── eval_closed_loop.py      # Policy evaluation in simulation
│   ├── parakeet_asr.py          # Voice transcription
│   ├── cosmos_reason.py         # Scene understanding
│   ├── cosmos_server.py         # HTTP server for Cosmos
│   ├── cosmos_client.py         # HTTP client for Cosmos
│   └── robot_pipeline.py        # Full Voice→Reason→Act
├── configs/
│   ├── reachy2_modality_config.py  # GR00T modality config
│   └── modality_reachy2.json       # Action space definition
├── Isaac-GR00T/                 # NVIDIA Isaac-GR00T (submodule)
├── libs/
│   └── reachy2_mujoco/          # Pollen Robotics simulator (submodule)
├── requirements.txt             # Python dependencies
└── README.md
```

**Note:** Checkpoints and datasets are downloaded separately from HuggingFace.

## Fine-Tuning GR00T for Your Robot

This section explains how to fine-tune GR00T N1.6 on your own dataset and add support for new robot embodiments.

### Adding a New Embodiment

GR00T uses "embodiment tags" to identify different robots. To add your own:

#### Step 1: Patch Isaac-GR00T

Apply the patch to add your embodiment tag:

```bash
cd Isaac-GR00T

# Apply the REACHY2 patch (or create your own)
patch -p1 < ../patches/add_reachy2_embodiment.patch
```

The patch modifies two files:

**`gr00t/data/embodiment_tags.py`** - Add enum entry:
```python
class EmbodimentTag(Enum):
    # ... existing tags ...

    REACHY2 = "reachy2"
    """
    The Pollen Robotics Reachy 2 humanoid robot.
    """
```

**`gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py`** - Add projector index:
```python
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    # ... existing mappings ...
    "new_embodiment": 10,
    "reachy2": 11,  # Use index 10-15 for custom robots
}
```

#### Step 2: Create Modality Config

Create a modality configuration that defines your robot's action space:

**`configs/reachy2_modality_config.py`**:
```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation,
    ActionType, ModalityConfig
)

register_modality_config(
    config={
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["front_cam"]  # Your camera key
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["arm_joints"]  # Your state key
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),  # Action horizon
            modality_keys=["arm_joints", "gripper"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="arm_joints"
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"]
        ),
    },
    embodiment_tag=EmbodimentTag.REACHY2,  # Your tag
)
```

### Collecting Training Data

Use the data collector to gather demonstrations:

```bash
# Collect 100 episodes with domain randomization
python scripts/data_collector.py \
    --episodes 100 \
    --output reachy2_dataset \
    --arm both \
    --randomize-object \
    --randomize-color \
    --cameras front_cam workspace_cam
```

Dataset format follows LeRobot v2.1:
```
reachy2_dataset/
├── meta/
│   ├── info.json           # Dataset metadata
│   ├── stats.json          # Normalization statistics
│   ├── tasks.jsonl         # Task descriptions
│   └── episodes.jsonl      # Episode info
├── data/chunk-000/
│   └── episode_*.parquet   # State/action data
└── videos/chunk-000/
    └── observation.images.*/
        └── episode_*.mp4   # Camera videos
```

### Training on Cloud/Cluster

#### Prerequisites

```bash
# Clone and setup Isaac-GR00T
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# Apply embodiment patch
patch -p1 < ../patches/add_reachy2_embodiment.patch

# Install dependencies
conda create -n groot python=3.10 -y
conda activate groot
pip install torch torchvision
pip install flash-attn --no-build-isolation
pip install -e .

# Login to HuggingFace
huggingface-cli login
```

#### Training Command

```bash
# Full training (2x A100, 30K steps)
python -m gr00t.train \
    --dataset_repo_id ganatrask/reachy2_100 \
    --embodiment_tag reachy2 \
    --video_backend decord \
    --num_gpus 2 \
    --batch_size 64 \
    --max_steps 30000 \
    --save_steps 3000 \
    --output_dir ./checkpoints/groot-reachy2
```

#### SLURM/Cluster Settings

| Resource | Quick Test | Full Training |
|----------|------------|---------------|
| CPU Cores | 8 | 32 |
| Memory | 32 GiB | 128 GiB |
| GPU | 1x A100 | 2x A100 |
| Wall Time | 1 hour | 6 hours |

### Loading Your Fine-Tuned Model

```python
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

# Load modality config first (must happen before policy init)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "modality_config",
    "configs/reachy2_modality_config.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Load policy
policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.REACHY2,
    model_path="checkpoints/groot-reachy2/checkpoint-30000",
    device="cuda",
    strict=True,
)

# Run inference
obs = {
    "video": {"front_cam": image[None, None, :, :, :]},
    "state": {"arm_joints": joints[None, None, :]},
    "language": {"annotation.human.task_description": [["Pick up the cube"]]},
}
action, _ = policy.get_action(obs)
```

### Troubleshooting

| Error | Solution |
|-------|----------|
| `REACHY2 not found` | Apply patch: `patch -p1 < patches/add_reachy2_embodiment.patch` |
| `Already registered` | Modality config loaded twice; add guard clause to prevent re-registration |
| `OOM on A100` | Reduce batch size: `--batch_size 32` |
| `torchcodec not available` | Use: `--video_backend decord` |
| `Duplicate enum key` | Re-clone Isaac-GR00T and apply patch once |
| `'reachy2' not in projector index` | Add mapping to `processing_gr00t_n1d6.py` (see patch) |

---

## Technical Highlights


### Performance Metrics

| Metric | Value |
|--------|-------|
| Dataset collection rate | ~2 episodes/min |
| GR00T inference | ~40ms/step (A100) |
| Cosmos reasoning | ~500ms/query |
| Parakeet transcription | 50x real-time |



## Links

| Resource | URL |
|----------|-----|
| GitHub | [ganatrask/NOVA](https://github.com/ganatrask/NOVA) |
| Model | [ganatrask/NOVA](https://huggingface.co/ganatrask/NOVA) |
| Dataset | [ganatrask/NOVA](https://huggingface.co/datasets/ganatrask/NOVA) |
| GR00T N1.6 | [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) |
| Cosmos Reason 2 | [nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2) |
| Parakeet ASR | [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) |
| reachy2_mujoco | [pollen-robotics/reachy2_mujoco](https://github.com/pollen-robotics/reachy2_mujoco) |
| LeRobot | [huggingface/lerobot](https://github.com/huggingface/lerobot) |

## Hardware Used

| Component | Specification |
|-----------|---------------|
| Training GPU | NVIDIA A100-SXM4-80GB |
| VRAM Usage | 44GB / 80GB |
| Training Time | ~6 hours (30K steps) |
| Inference | Works on RTX 5090 |

## Acknowledgments

- **[Pollen Robotics](https://www.pollen-robotics.com/)** - Reachy 2 humanoid robot & MuJoCo simulation
- **[NVIDIA](https://developer.nvidia.com/)** - GR00T N1.6, Cosmos Reason 2, Parakeet ASR
- **[HuggingFace](https://huggingface.co/)** - LeRobot framework & model hosting
- **[DeepMind](https://mujoco.org/)** - MuJoCo physics engine
- **[VESSL AI](https://vessl.ai/)** - GPU compute credits for model training

## License

This project uses NVIDIA models under their respective licenses:

- **NVIDIA GR00T N1.6**: [NVIDIA Open Model License](https://developer.nvidia.com/open-model-license)
- **NVIDIA Cosmos Reason 2**: [License](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/LICENSE)
- **NVIDIA Parakeet ASR**: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)


