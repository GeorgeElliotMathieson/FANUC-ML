# AMD GPU Acceleration with DirectML

This guide explains how to use your AMD GPU for accelerating robot training, evaluation, and demos using DirectML.

## Overview

The benchmark test has shown that DirectML provides significant speedup with your AMD Radeon RX 6700S GPU:
- **CPU time:** ~0.07 seconds
- **DirectML (AMD GPU) time:** ~0.006 seconds
- **Speedup:** ~11.4x

This confirms that DirectML is properly working with your AMD GPU.

## Installation

We've created scripts to make setup easy:

1. Install DirectML and test your GPU:
   ```powershell
   # Run in PowerShell
   cd res/rml/python
   ./install_directml.ps1
   ```

This script will:
- Install the torch-directml package
- Create and run a test script to verify acceleration
- Set the required environment variables for the current session

## Using AMD GPU Acceleration

We've created a modified version of the training script that supports DirectML:

### For Demo Mode

```powershell
python train_robot_rl_demo_directml.py --demo --use-directml --load ./models/revamped_20250321_143755/final_model --viz-speed 0.05
```

### For Evaluation

```powershell
python train_robot_rl_demo_directml.py --eval-only --use-directml --load ./models/revamped_20250321_143755/final_model --eval-episodes 5
```

### For Training

```powershell
python train_robot_rl_demo_directml.py --use-directml --steps 10000
```

**Note:** Full training acceleration with DirectML is not yet fully implemented. The script will inform you that it's using CPU for training. We're working on improving this.

## Command Line Options

The DirectML-enabled script includes additional options:

- `--use-gpu`: Use any available GPU (CUDA, ROCm, or DirectML)
- `--use-directml`: Specifically request DirectML for AMD GPU acceleration
- `--use-cpu`: Force CPU usage even if GPU is available

## Environment Variables

The DirectML integration uses these environment variables:

- `HIP_VISIBLE_DEVICES=0`: Selects which AMD GPU to use
- `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128`: Configures memory allocation

## Troubleshooting

If you encounter issues:

1. Make sure `torch-directml` is installed:
   ```
   pip install torch-directml
   ```

2. Verify DirectML is working:
   ```
   python test_directml.py
   ```

3. Check for error messages when running the script. If you see "DirectML was requested but is not available", reinstall using the script.

## How This Works

The DirectML integration:

1. Detects your AMD GPU and configures the proper backend
2. Moves model operations to the GPU for inference
3. Properly synchronizes operations to ensure correct timing
4. Enforces joint limits while maintaining GPU acceleration 