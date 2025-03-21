# Robot Training with DirectML Acceleration for AMD GPUs

This documentation explains how to use the DirectML-enabled robot training script for AMD GPUs. The script provides GPU acceleration for training robot reinforcement learning models using PyTorch and DirectML.

## Requirements

- Windows 10/11
- Python 3.8 or higher
- PyTorch 2.0 or higher
- torch-directml package
- AMD GPU with recent drivers

## Installation

1. Install PyTorch:
   ```
   pip install torch
   ```

2. Install DirectML for PyTorch:
   ```
   pip install torch-directml
   ```

3. Install other dependencies:
   ```
   pip install stable-baselines3 gymnasium pybullet numpy psutil
   ```

## Usage

The script supports various modes of operation, including training, evaluation, and demo modes. Here are some example commands:

### Training

```bash
# Basic training with DirectML acceleration
python train_robot_rl_demo_directml.py --steps 50000 --parallel 4

# Force CPU usage even with AMD GPU available
python train_robot_rl_demo_directml.py --steps 50000 --use-cpu

# Adjust memory limit for large networks
python train_robot_rl_demo_directml.py --steps 50000 --memory-limit 256
```

### Evaluation

```bash
# Evaluate a trained model
python train_robot_rl_demo_directml.py --eval-only --load ./models/your_model_folder/final_model --eval-episodes 20
```

### Demo

```bash
# Visualize a trained model
python train_robot_rl_demo_directml.py --demo --load ./models/your_model_folder/final_model --viz-speed 0.02

# Create a video of the demo
python train_robot_rl_demo_directml.py --demo --load ./models/your_model_folder/final_model --save-video
```

## Command Line Arguments

### Basic Options
- `--steps`: Total number of training steps (default: 300000)
- `--load`: Path to a pre-trained model to continue training
- `--algorithm`: RL algorithm to use (choices: 'ppo', 'sac', 'td3', default: 'ppo')

### Evaluation Options
- `--eval-only`: Only evaluate the model, don't train
- `--eval-episodes`: Number of episodes for evaluation (default: 20)

### Demo Options
- `--demo`: Run a demonstration of the model
- `--save-video`: Save a video of the demonstration
- `--viz-speed`: Visualization speed (delay in seconds, default: 0.02)

### Training Options
- `--parallel`: Number of parallel environments (default: 8)
- `--parallel-viz`: Enable visualization for parallel environments
- `--gui`: Enable GUI visualization (default: True)
- `--no-gui`: Disable GUI visualization
- `--learning-rate`: Learning rate for the optimizer (default: 0.001)
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable verbose output (default: True)
- `--strict-limits`: Strictly enforce joint limits from URDF

### GPU Options
- `--use-cpu`: Force CPU usage even if GPU is available
- `--disable-directml`: Disable DirectML acceleration
- `--sync-freq`: How often to synchronize with DirectML device (default: 10)
- `--memory-limit`: Memory limit for DirectML operations in MB (default: 128)

## Troubleshooting

### Memory Issues
If you encounter out-of-memory errors, try reducing the `--memory-limit` value:
```
python train_robot_rl_demo_directml.py --memory-limit 64
```

### Performance Issues
If you notice poor performance:
1. Make sure your GPU drivers are up to date
2. Try using fewer parallel environments with `--parallel 1`
3. Disable visualization with `--no-gui`
4. Increase synchronization frequency with `--sync-freq 20`

### Compatibility
DirectML is still evolving and may not support all PyTorch operations natively. Some operations may fall back to CPU, which can impact performance. For best results, use simple network architectures.

## Environment Variables

The script sets these environment variables automatically, but you can also set them manually:

- `PYTORCH_HIP_ALLOC_CONF`: Configure memory allocation behavior
- `HIP_VISIBLE_DEVICES`: Control which GPUs are used by DirectML

## Technical Details

The script works by monkey-patching PyTorch's device handling functions to redirect operations to the DirectML device. This allows existing code and libraries (like stable-baselines3) to run on AMD GPUs without code modifications.

Performance will vary depending on your specific GPU model, driver version, and the operations used in your neural networks. 