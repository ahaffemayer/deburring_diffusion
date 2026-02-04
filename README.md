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

4. **Verify installation**:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   
   # Test cuRobo
   from curobo.types.base import TensorDeviceType
   print("cuRobo loaded successfully!")
   ```

## Project Structure

```
deburring_diffusion/
├── .devcontainer/
│   ├── devcontainer.json      # VS Code container configuration
│   └── Dockerfile             # Container build instructions
├── examples/
│   └── traj_generator_multiple_configurations_to_multiple_poses.py
│                              # Trajectory generation examples
├── scripts/
│   ├── train.sh               # Training launcher
│   └── config/
│       └── diffusion_config.yml  # Training configuration
├── results/                   # Training outputs and checkpoints
└── README.md
```

## Usage

### Generate Trajectories

Run the trajectory generator to create motion plans:

```bash
python examples/traj_generator_multiple_configurations_to_multiple_poses.py
```

This script demonstrates how to generate collision-free trajectories from multiple robot configurations to target poses using cuRobo.

### Train the Diffusion Model

1. **Configure training** (optional):
   Edit `scripts/config/diffusion_config.yml` to adjust:
   - Dataset path
   - Number of epochs

2. **Launch training**:
   ```bash
   bash scripts/train.sh
   ```

3. **Monitor training**:
   - TensorBoard: Open http://localhost:6006 in your browser
   - Results are saved to `/results/`

### Visualization

- **TensorBoard**: Automatically forwarded on port 6006
- **Meshcat** (for 3D visualization): Available on port 7000

## Configuration

### Training Configuration

Edit `scripts/config/diffusion_config.yml`:

```yaml
# Example configuration
data:
  data_file: /workspaces/deburring_diffusion/results/traj_generator/multiple_configuration_multiple_poses.json
trainer:
  max_epochs: 1000
  default_root_dir: /workspaces/deburring_diffusion/results/diffusion
```

### Hardware Configuration

The container is optimized for RTX 5080 (Compute Capability 12.0):
- `TORCH_CUDA_ARCH_LIST="12.0+PTX"` for forward compatibility
- PyTorch Nightly with CUDA 12.8 support
- 8GB shared memory for efficient data loading

## Resources

- **cuRobo Documentation**: https://curobo.org/
- **cuRobo Examples**: `/workspace/curobo/examples/`
- **PyTorch Nightly**: https://pytorch.org/get-started/locally/

## Citation

If you use this work in your research, please cite:

```bibtex
[Add your citation here]
```
