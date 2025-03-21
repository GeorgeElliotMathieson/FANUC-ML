# AMD GPU Usage Guide for Robot Training

This guide provides detailed instructions for configuring your AMD GPU to accelerate robot training with PyTorch. Your AMD Radeon RX 6700S GPU can significantly speed up training when properly configured.

## Diagnosis

Running the diagnostic script, we've identified that:

1. You have an AMD Radeon RX 6700S GPU with 4GB VRAM
2. Your current PyTorch installation (2.6.0+cpu) is CPU-only
3. To use GPU acceleration, you need to install PyTorch with ROCm support

## Solutions

We've provided two solutions to enable AMD GPU usage:

### Quick Fix (Recommended)

We've created scripts that automate the installation process:

1. **Install PyTorch with ROCm support**:
   ```
   # Using PowerShell:
   .\install_amd_pytorch.ps1
   
   # Or using Batch:
   install_amd_pytorch.bat
   ```

2. **Set permanent environment variables** (run as Administrator):
   ```
   .\set_permanent_amd_vars.ps1
   ```

3. **Test GPU performance**:
   ```
   python test_amd_gpu_performance.py
   ```

4. **Train with AMD GPU acceleration**:
   ```
   python train_robot_rl_demo.py --steps 10000 --parallel 8 --use-amd
   ```

### Manual Installation

If you prefer to perform the steps manually:

1. **Uninstall current PyTorch**:
   ```
   pip uninstall -y torch torchvision torchaudio
   ```

2. **Install PyTorch with ROCm support**:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
   ```

3. **Set environment variables for current session**:
   ```
   # In PowerShell:
   $env:HIP_VISIBLE_DEVICES = "0"
   $env:PYTORCH_HIP_ALLOC_CONF = "max_split_size_mb:128"
   
   # In Command Prompt:
   set HIP_VISIBLE_DEVICES=0
   set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
   ```

4. **Set permanent environment variables**:
   - Open System Properties (right-click on This PC → Properties → Advanced System Settings)
   - Click on "Environment Variables"
   - Add new System variables:
     - HIP_VISIBLE_DEVICES = 0
     - PYTORCH_HIP_ALLOC_CONF = max_split_size_mb:128

## Performance Optimization

To maximize AMD GPU performance:

1. **Increase parallel environments**:
   - Use `--parallel 8` or higher when training
   - This utilizes the GPU more efficiently

2. **Adjust batch size** if you encounter memory issues:
   - The `max_split_size_mb:128` setting helps with memory fragmentation
   - If training fails with out-of-memory errors, try decreasing the parallel count

3. **Verify GPU usage**:
   - Run the performance test to confirm speedup:
     ```
     python test_amd_gpu_performance.py
     ```
   - You should see significant speedup compared to CPU (5-20x depending on the operation)

## Troubleshooting

**PyTorch doesn't detect the AMD GPU**:
- Verify you have the correct PyTorch version with `pip list | findstr torch`
- The version should NOT have "+cpu" in it
- Environment variables must be set correctly

**Training is still slow**:
- Run the performance test to verify GPU acceleration is working
- Check if other applications are using the GPU
- Monitor GPU usage using Windows Task Manager

**Out of memory errors**:
- Reduce the number of parallel environments
- The RX 6700S has 4GB VRAM, so be mindful of memory constraints

## Technical Notes

1. **ROCm support on Windows**:
   - ROCm support for Windows is still evolving
   - We're using the official PyTorch binaries with ROCm 5.6 support

2. **Environment variables**:
   - `HIP_VISIBLE_DEVICES=0` selects the primary AMD GPU
   - `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` optimizes memory allocation

3. **Verification**:
   - You can verify GPU detection with:
     ```python
     import torch
     print(f"ROCm/HIP available: {hasattr(torch, 'hip') and torch.hip.is_available()}")
     print(f"Device count: {torch.hip.device_count() if hasattr(torch, 'hip') and hasattr(torch.hip, 'device_count') else 0}")
     ```

## Further Resources

- [PyTorch ROCm Installation Guide](https://pytorch.org/get-started/locally/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/en/latest/)
- For further assistance, run the diagnostic script:
  ```
  python amd_gpu_fix.py
  ``` 