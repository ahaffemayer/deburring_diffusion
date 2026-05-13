# Deburring Diffusion

A diffusion-based approach for robot trajectory generation in deburring tasks, built with PyTorch and NVIDIA cuRobo for GPU-accelerated motion planning.

## Overview

This project uses diffusion models to generate robot trajectories for deburring operations. It leverages cuRobo for efficient collision-free motion generation and PyTorch for training the diffusion model.

## What's Included

- **PyTorch Nightly** with CUDA 12.8 support for RTX 5080 (SM 12.0)
- **cuRobo**: NVIDIA's CUDA-accelerated robot motion generation library
- **Development tools**: Python, Pylance, Ruff (linting/formatting)
- **Jupyter support**: Full notebook integration in VS Code
- **Visualization**: TensorBoard (port 6006) and Meshcat (port 7000)
- **Common packages**: numpy, pandas, matplotlib, scikit-learn

## Quick Start

### Prerequisites

- NVIDIA GPU with Compute Capability 12.0+ (RTX 5080 or newer)
- Docker with NVIDIA Container Toolkit installed
- VS Code with Dev Containers extension

### Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:ahaffemayer/deburring_diffusion.git
   cd deburring_diffusion
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Start the dev container**:
   - Press `F1` → "Dev Containers: Reopen in Container"
   - **First build takes ~30-40 minutes** (building PyTorch for SM 12.0 + cuRobo)
   - Subsequent starts are much faster (~10-15 seconds)

4. **Install this package in editable mode**:
   ```bash
   pip install -e .
   ```

   This makes local imports such as `deburring_diffusion.robot.curobo_utils` work from
   the example scripts.

5. **Verify installation**:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   
   # Test cuRobo
   from curobo.types.base import TensorDeviceType
   print("cuRobo loaded successfully!")
   ```

## Usage

### Generate Tomato Pick-Place Training Data

The tomato pick-place generator creates collision-aware trajectories for
`tomato_reach_scene`, where the Franka picks `B_tomato` and places it on the
upper shelf while avoiding the other objects and shelf.

It supports:
- multiple sampled scenes
- random place poses on the upper shelf
- random initial robot configurations
- multiple joint-space modes for the same fixed scene/place target
- cuRobo batched planning on the GPU

Recommended training-size run:

```bash
python examples/traj_generator_tomato_pick_place_batched_training.py \
  --n-scenes 50 \
  --places-per-scene 4 \
  --modalities-per-place 4 \
  --batch-size 32 \
  --max-candidates-per-target 128 \
  --min-mode-rms-rad 0.10 \
  --start-noise-rad 0.35 \
  --trajectory-length 100 \
  --output results/traj_generator/tomato_pick_place_training.json
```

This targets up to:

```text
50 scenes x 4 place poses x 4 modes = 800 trajectories
```

Notes:
- `--min-mode-rms-rad` filters out near-duplicate joint-space solutions for the same scene/place target.
- `Batch mode enable graph is only supported with num_graph_seeds==1` is a cuRobo warning. If trajectories are saved, it is safe to ignore.

### Visualization

- **Meshcat** (for 3D visualization): Available on port 7000

Visualize a generated tomato pick-place trajectory:

```bash
python examples/visualization/visualize_tomato_pick_place_meshcat.py \
  --dataset results/traj_generator/tomato_pick_place_training.json \
  --index 0 \
  --host 0.0.0.0 \
  --web-port 7000 \
  --loop
```

Open:

```text
http://localhost:7000/static/
```
