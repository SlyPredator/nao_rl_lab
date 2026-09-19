# Aldebaran NAO Manager-Based Locomotion

This repository contains the standalone, lean **Manager-Based** IsaacLab implementation for the Aldebaran NAO humanoid robot locomotion environments.

## Prerequisites

IsaacLab (and its PhysX / rsl-rl backend) runs on a CUDA GPU. This repository is tested and verified with:

| Component     | Version |
|---------------|---------|
| OS            | Ubuntu 24.04 LTS |
| Python        | 3.11.15 |
| Isaac Sim     | 5.1.0 |
| IsaacLab      | 2.3.2 |
| rsl-rl-lib    | 5.0.1 |
| PyTorch       | 2.7.0 (CUDA 12.8) |
| Gymnasium     | 1.2.1 |

Install IsaacLab by following its official documentation: https://isaac-sim.github.io/IsaacLab

## Installation

Register the package into your Python / Isaac Sim environment by running:

```bash
cd nao_rl_lab
pip install -e .
```

## Environments

The following environments are registered via `gymnasium` and available for training:

- **`Nao-Velocity-Flat-v0`**: Tuned omnidirectional flat-ground locomotion environment with push perturbations, friction randomization, and yaw-frame velocity tracking.
- **`Nao-Velocity-Flat-Play-v0`**: Evaluation variant of flat ground optimized for visualization and deployment benchmarking.
- **`Nao-Velocity-Rough-v0`**: Locomotion on gentle rough terrain (mild roughness, gentle slopes, and flat patches).
- **`Nao-Velocity-Rough-Play-v0`**: Evaluation variant for rough terrain visualization.

## Training & Evaluation

### Training
Train the locomotion policy using RSL-RL:

```bash
# Flat ground training
python scripts/train.py --task Nao-Velocity-Flat-v0 --num_envs 4096 --max_iterations 3000

# Rough terrain training
python scripts/train.py --task Nao-Velocity-Rough-v0 --num_envs 4096 --max_iterations 3000
```

### Playback & Visualization
Visualize a trained policy directly inside Isaac Sim:

```bash
# Playback flat ground policy
python scripts/play.py --task Nao-Velocity-Flat-Play-v0 --checkpoint logs/rsl_rl/nao_tuned_flat/<RUN_TIMESTAMP>/model_2100.pt

# Playback rough terrain policy
python scripts/play.py --task Nao-Velocity-Rough-Play-v0 --checkpoint logs/rsl_rl/nao_rough/<RUN_TIMESTAMP>/model_3000.pt
```
