# Robot Positioning with Reinforcement Learning

This repository contains implementations for training a FANUC robot arm to position its end-effector at target locations using reinforcement learning.

## Implementations

Two different implementations are included:

1. **Original Implementation** (`train_robot_rl_positioning.py`): 
   - Uses velocity control for the robot's joints
   - SAC (Soft Actor-Critic) algorithm
   - Reward based on relative progress toward the target
   - Monolithic neural network architecture

2. **Revamped Implementation** (`train_robot_rl_positioning_revamped.py`): 
   - Uses delta joint position control for more precise movements
   - PPO (Proximal Policy Optimization) algorithm with observation/reward normalization
   - Sophisticated multi-component reward function
   - Enhanced neural network with structured feature extraction and residual connections
   - Temporal information with state history
   - Curriculum learning and advanced target sampling strategies

For detailed information about the revamped approach, see [README_REVAMPED.md](./README_REVAMPED.md).

## Installation

### Requirements

- Python 3.7+
- PyBullet
- PyTorch
- Stable-Baselines3
- NumPy
- Matplotlib

Install the required packages:

```bash
pip install torch numpy gymnasium stable-baselines3 pybullet matplotlib
```

## Usage

### Original Implementation

Train a new model:
```bash
python train_robot_rl_positioning.py --parallel 4 --steps 500000
```

Evaluate a trained model:
```bash
python train_robot_rl_positioning.py --eval-only --load ./models/sac_xyz/final_model
```

### Revamped Implementation

Train a new model:
```bash
python train_robot_rl_positioning_revamped.py --parallel 8 --steps 1000000
```

Evaluate a trained model:
```bash
python train_robot_rl_positioning_revamped.py --eval-only --load ./models/revamped_xyz/final_model
```

Run a demonstration:
```bash
python train_robot_rl_positioning_revamped.py --demo --load ./models/revamped_xyz/final_model --viz-speed 0.05
```

## Comparing the Approaches

The revamped approach addresses several key limitations of the original implementation:

1. **Action Representation**: 
   - Original: Uses velocity control which is difficult to control precisely
   - Revamped: Uses delta joint positions for finer control and smoother movements

2. **Reward Function**:
   - Original: Simple progress-based reward with caps
   - Revamped: Sophisticated multi-component reward with distance, progress, efficiency, smoothness, and joint limit components

3. **Neural Network**:
   - Original: Large but simple network with standard layers
   - Revamped: Structured feature extraction and residual connections for better learning

4. **Observation Space**:
   - Original: Basic observation with current state only
   - Revamped: Rich observation including history of states for temporal awareness

5. **Learning Algorithm**:
   - Original: SAC (good for exploration but can be unstable)
   - Revamped: PPO with normalization (more stable and sample-efficient)

## Performance Comparison

The revamped approach significantly outperforms the original implementation in:
- Success rate in reaching targets
- Movement efficiency (fewer steps to reach targets)
- Smoothness of motion
- Generalization to new target positions
- Learning speed and stability

## Advanced Options

Both implementations support various command-line arguments:
- Parallel training environments
- CUDA acceleration
- Visualization options
- Logging and monitoring

For a full list of options, run either script with the `--help` flag.

## Implementation Details

Both implementations use:
- PyBullet for physics simulation
- Stable-Baselines3 for reinforcement learning algorithms
- A FANUC LR Mate 200iC robot model

The environment simulates:
- Joint limits and dynamics
- Realistic end-effector positioning
- Target visualization and workspace constraints

## Development

To extend this work, consider:
1. Adding obstacle avoidance capabilities
2. Implementing multi-goal training
3. Incorporating real robot hardware for sim-to-real transfer
4. Experimenting with different RL algorithms like TD3 or DDPG 