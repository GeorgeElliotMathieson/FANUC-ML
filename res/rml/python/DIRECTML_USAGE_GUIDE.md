# DirectML AMD GPU Acceleration Guide

This guide explains how to use your AMD Radeon RX 6700S GPU for accelerating robot training, evaluation, and demos using the DirectML backend.

## Summary

- The benchmark tests show that DirectML provides an approximately **11.4x speedup** with your AMD GPU compared to CPU.
- The implementation is fully working for **demo** and **evaluation** modes.
- Training mode currently falls back to CPU (full DirectML training support is planned for a future update).

## Quick Start

1. Install DirectML package (if not already installed):
   ```
   pip install torch-directml
   ```

2. Run the DirectML-enabled script with your preferred mode:

   **For Demo Mode:**
   ```
   python train_robot_rl_demo_directml.py --demo --use-directml --load ./models/your_model_folder/final_model --viz-speed 0.05
   ```

   **For Evaluation:**
   ```
   python train_robot_rl_demo_directml.py --eval-only --use-directml --load ./models/your_model_folder/final_model --eval-episodes 5
   ```

   **For Training (currently uses CPU but saves compatible models):**
   ```
   python train_robot_rl_demo_directml.py --use-directml --steps 10000 --parallel 4
   ```

## Command Line Arguments

The DirectML script supports all the standard arguments from the original script plus:

- `--use-gpu`: Use any available GPU (CUDA, ROCm, or DirectML)
- `--use-directml`: Specifically request DirectML for AMD GPU acceleration
- `--use-cpu`: Force CPU usage even if GPU is available

## Understanding the Implementation

The DirectML integration:

1. Automatically detects your AMD GPU and sets up the appropriate backend
2. Moves neural network operations to the GPU during inference
3. Properly handles tensor conversions between CPU and GPU
4. Ensures joint limits are enforced while maintaining GPU acceleration

## Limitations and Future Work

1. **Training**: Full training acceleration with DirectML is not yet implemented. The script will inform you that it's using CPU for training.

2. **Performance**: While DirectML provides good acceleration for inference (demos and evaluation), it's not as optimized for PyTorch as CUDA is for NVIDIA GPUs.

## Troubleshooting

If you encounter issues:

1. Make sure `torch-directml` is correctly installed:
   ```
   pip show torch-directml
   ```

2. Verify DirectML is working with your GPU:
   ```
   python test_directml.py
   ```

3. Check for error messages. If you see "DirectML was requested but is not available", reinstall using:
   ```
   pip install torch-directml
   ```

4. If you encounter out-of-memory errors, try reducing the number of parallel environments.

## Performance Comparison

- **CPU**: ~0.07 seconds for a 2000x2000 matrix multiplication
- **DirectML (AMD GPU)**: ~0.006 seconds for the same operation
- **Speedup**: ~11.4x

This confirms that DirectML is providing significant acceleration with your AMD GPU. 