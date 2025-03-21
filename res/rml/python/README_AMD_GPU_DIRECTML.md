# Robot Reinforcement Learning with AMD GPU Acceleration

This documentation explains how to use AMD GPU acceleration with DirectML for robot reinforcement learning training and inference.

## System Requirements

- Windows 10/11
- AMD GPU with recent drivers installed
- Python 3.8+ (3.11 recommended)
- PyTorch 2.0.0+ (CPU version)
- torch-directml package

## Installation

1. Ensure you have the CPU version of PyTorch installed:
   ```
   pip install torch torchvision torchaudio
   ```

2. Install the DirectML package:
   ```
   pip install torch-directml
   ```

3. Install other required dependencies:
   ```
   pip install stable-baselines3 gymnasium psutil
   ```

## Quick Start

### Method 1: Python Launcher (Recommended)

The simplest way to use AMD GPU acceleration is through the Python launcher script:

```
python run_amd_gpu_training.py [mode] [options]
```

This script handles all environment variables and settings automatically.

#### Available Modes:

- `--train`: Train a new model (default)
- `--demo`: Run demo with existing model 
- `--eval`: Evaluate existing model

#### Options:

- `--steps N`: Number of training steps (default: 10000)
- `--parallel N`: Number of parallel environments (default: 8)
- `--model PATH`: Path to model (required for --demo and --eval)
- `--episodes N`: Number of evaluation episodes (default: 5)
- `--viz-speed N`: Visualization speed for demo (default: 0.05)
- `--save-video`: Save video of demo/evaluation
- `--install-directml`: Install torch-directml if not available

#### Examples:

```
# Train a new model with 20,000 steps and 8 parallel environments
python run_amd_gpu_training.py --train --steps 20000 --parallel 8

# Run a demo using a pre-trained model
python run_amd_gpu_training.py --demo --model ./models/revamped_20250321_190817/final_model

# Evaluate a model for 10 episodes
python run_amd_gpu_training.py --eval --model ./models/revamped_20250321_190817/final_model --episodes 10

# First-time setup (will install DirectML if needed)
python run_amd_gpu_training.py --install-directml
```

### Method 2: Batch File

Alternatively, you can use the provided batch file:

```
train_with_amd_gpu_optimized.bat [mode] [options]
```

This batch file sets appropriate environment variables and calls the Python script.

#### Available Modes:

- `--train`: Train a new model (default)
- `--demo`: Run demo with existing model 
- `--eval`: Evaluate existing model

#### Options:

- `--steps N`: Number of training steps (default: 10000)
- `--parallel N`: Number of parallel environments (default: 8)
- `--model PATH`: Path to model (required for --demo and --eval)
- `--episodes N`: Number of evaluation episodes (default: 5)
- `--viz-speed N`: Visualization speed for demo (default: 0.05)

#### Examples:

```
# Train a new model with 20,000 steps and 8 parallel environments
train_with_amd_gpu_optimized.bat --train --steps 20000 --parallel 8

# Run a demo using a pre-trained model
train_with_amd_gpu_optimized.bat --demo --model ./models/revamped_20250321_190817/final_model

# Evaluate a model for 10 episodes
train_with_amd_gpu_optimized.bat --eval --model ./models/revamped_20250321_190817/final_model --episodes 10
```

### Method 3: Manual Usage

If you prefer to run the Python script directly:

```
python train_robot_rl_demo_directml.py [options]
```

This requires manually setting environment variables for optimal performance.

#### Key Command Line Arguments:

- `--steps`: Total number of timesteps for training
- `--parallel`: Number of environments to run in parallel
- `--demo`: Run in demo mode with visualization
- `--eval-only`: Run in evaluation mode only
- `--load`: Path to a saved model to load
- `--eval-episodes`: Number of episodes to run during evaluation
- `--viz-speed`: Speed of visualization (lower is slower)

## How It Works

The implementation uses several techniques to accelerate robot reinforcement learning using AMD GPUs:

1. **DirectML Integration**: The script uses torch-directml to enable GPU acceleration on AMD hardware.

2. **Function Patching**: Key PyTorch functions are patched to redirect operations to DirectML, allowing frameworks like stable-baselines3 to utilize the GPU without modifications.

3. **Memory Management**: The script dynamically adjusts the batch size and number of parallel environments based on available GPU memory.

4. **Performance Monitoring**: Training progress and memory usage are monitored and reported at regular intervals.

5. **Synchronization Handling**: Special handling for synchronization issues ensures accurate performance measurements.

## Performance Expectations

With an AMD Radeon RX 6700S GPU, you can expect:
- Up to 10x speedup compared to CPU-only training
- Ability to train larger models or with more parallel environments
- Faster inference during evaluation and demo modes

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:
1. Reduce the number of parallel environments with `--parallel`
2. Decrease the batch size (internally managed)
3. Make sure no other GPU-intensive applications are running

### DirectML Not Available

If you see a warning that DirectML is not available:
1. Install torch-directml with `pip install torch-directml`
2. Ensure your AMD GPU drivers are up to date
3. Verify that your GPU is compatible with DirectML
4. Use the `--install-directml` flag with the Python launcher

### Performance Issues

If performance is lower than expected:
1. Check if other applications are using the GPU
2. Monitor GPU utilization with Task Manager
3. Try adjusting the environment variables in the batch file

## Environment Variables

The following environment variables are automatically set by the scripts but can be adjusted for performance tuning:

- `HIP_VISIBLE_DEVICES`: Controls which GPU is used (default: 0)
- `PYTORCH_HIP_ALLOC_CONF`: Controls memory allocation behavior
- `DML_PREFETCH_BUFFERS`: Controls buffer prefetching for operations
- `DML_THREAD_COUNT`: Controls number of threads for DirectML operations

## Advanced Configuration

For advanced users, you can modify the following aspects:

1. **Memory Limits**: The script is configured to use at most 40% of the available VRAM. This can be adjusted in the code.

2. **Thread Count**: DirectML's thread count is set to half of the available logical cores by default.

3. **Custom Models**: You can load and evaluate custom models by specifying the path with the `--model` parameter.

## Limitations

- DirectML support is currently experimental and may not support all PyTorch operations
- Some operations may still fall back to CPU execution
- Performance may vary depending on the specific AMD GPU model and driver version

## References

- [torch-directml GitHub](https://github.com/microsoft/DirectML/tree/master/PyTorch/torch-directml)
- [DirectML Documentation](https://docs.microsoft.com/en-us/windows/ai/directml/dml)
- [stable-baselines3 Documentation](https://stable-baselines3.readthedocs.io/) 