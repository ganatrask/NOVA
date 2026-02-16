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
│   │    ASR      │   text        │  "I see a red, white   │               │
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

# Clone repository
git clone https://github.com/ganatrask/NOVA.git
cd NOVA
```

### Installation
```bash
# Activate main environment
conda activate reachy_groot

# Install reachy2_mujoco
pip install -e libs/reachy2_mujoco

# Install dependencies
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
│   ├── data_collector.py        # Dataset collection (600+ LOC)
│   ├── eval_closed_loop.py      # Policy evaluation in simulation
│   ├── parakeet_asr.py          # Voice transcription
│   ├── cosmos_reason.py         # Scene understanding
│   ├── cosmos_server.py         # HTTP server for Cosmos
│   └── robot_pipeline.py        # Full Voice→Reason→Act
├── libs/
│   └── reachy2_mujoco/          # Pollen Robotics simulator
├── checkpoints/
│   └── groot-reachy2-pickplace/ # Trained GR00T models (85GB)
├── dataset_100/                 # LeRobot dataset (198MB)
└── docs/
    └── CLAUDE.md                # Development notes (72KB)
```

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


