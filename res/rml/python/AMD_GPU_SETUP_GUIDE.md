# AMD GPU Setup Guide for Robot Training

This guide explains how to set up your AMD Radeon RX 6700S GPU for accelerated training of robot positioning models.

## Requirements

- AMD Radeon RX 6700S GPU
- PyTorch with ROCm/HIP support
- Windows 10/11 or Linux with ROCm drivers installed

## Quick Start

For the fastest way to get started:

1. Install PyTorch with ROCm support (see below)
2. Run the PowerShell script to set environment variables:
   ```
   .\setup_amd_gpu_env.ps1
   ```
3. Train with AMD GPU acceleration:
   ```
   python train_robot_rl_demo.py --use-amd --steps 10000 --parallel 8
   ```

Alternatively, you can simply double-click the `train_with_amd_gpu.bat` file to start training with AMD GPU support.

## Installing PyTorch with ROCm Support

### Option 1: Use the Helper Script

We've provided a helper script to guide you through the installation process:

```
python install_amd_gpu_support.py
```

This interactive script will:
- Check your system compatibility
- Detect AMD GPU hardware
- Verify ROCm installation
- Guide you through PyTorch installation
- Configure environment variables
- Test your setup

### Option 2: Manual Installation (Windows)

1. Visit [PyTorch's website](https://pytorch.org/get-started/locally/)
2. Download the appropriate ROCm-enabled wheel for your Python version
3. Install the wheel using pip:
   ```
   pip install [downloaded-wheel-file]
   ```
4. Verify installation:
   ```
   python -c "import torch; print(torch.version.hip)"
   ```

### Option 3: Manual Installation (Linux)

1. Install the ROCm drivers for your distribution
2. Install PyTorch with ROCm support:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
   ```
3. Verify installation:
   ```
   python -c "import torch; print(torch.version.hip)"
   ```

## Environment Variables

For optimal performance, set these environment variables:

- `HIP_VISIBLE_DEVICES=0` - Use the first AMD GPU
- `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` - Optimize memory allocation

These are automatically set by the provided scripts.

## Command-Line Options

The training script includes several options for GPU control:

- `--use-gpu`: Use any available GPU (NVIDIA or AMD)
- `--use-amd`: Specifically target AMD GPU with ROCm/HIP
- `--use-cpu`: Force CPU usage even if GPU is available
- `--parallel N`: Use N parallel environments (recommended: 8-16 with GPU)

## Performance Tips

1. **Increase Parallelism**: With a GPU, you can significantly increase the number of parallel environments:
   ```
   python train_robot_rl_demo.py --use-amd --parallel 16
   ```

2. **Disable GUI**: For pure training performance, disable the GUI:
   ```
   python train_robot_rl_demo.py --use-amd --no-gui --parallel 16
   ```

3. **Memory Optimization**: If you encounter memory issues, try reducing batch sizes or the number of parallel environments.

## Troubleshooting

- **PyTorch doesn't detect AMD GPU**: Ensure you have PyTorch with ROCm support installed
- **Out of memory errors**: Reduce parallel environments or adjust memory allocation
- **Performance issues**: Experiment with different parallelism levels

## Windows-Specific Notes

The current state of ROCm/HIP support on Windows is evolving. If you encounter issues:

1. Consider dual-booting with Linux for better ROCm support
2. Try the CPU version if ROCm support is unstable
3. Check the AMD and PyTorch documentation for the latest Windows support updates

## Further Resources

- [AMD ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

For any issues with the robot training specifically, refer to the README_DEMO.md file. 