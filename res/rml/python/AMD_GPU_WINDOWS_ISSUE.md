# AMD GPU Support on Windows: Known Issues and Solutions

## Current Situation

We've identified that the official PyTorch ROCm wheels are not currently available for Windows. This is a limitation in the PyTorch distribution system rather than a problem with your setup or our scripts.

According to the [PyTorch website](https://pytorch.org/get-started/locally/), ROCm packages are officially provided for Linux systems only, not for Windows. This explains why our installation commands are failing.

## Alternative Solutions

Here are several alternatives to get your AMD GPU working with PyTorch for robot training:

### Option 1: Use DirectML Support (Recommended for Windows)

Microsoft provides DirectML backend for PyTorch, which can work with AMD GPUs on Windows:

```powershell
# Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with DirectML support
pip install torch-directml
```

After installation, you can use DirectML with:

```python
import torch_directml
dml = torch_directml.device()
tensor = torch.tensor([1, 2, 3], device=dml)
```

Our training script would need to be modified to use DirectML instead of ROCm.

### Option 2: Use WSL2 (Windows Subsystem for Linux)

For the best ROCm support:

1. Install WSL2 with Ubuntu
2. Install ROCm drivers in WSL2
3. Install PyTorch with ROCm support in the Linux environment
4. Run the training script in WSL2

### Option 3: Use CPU Training

While not as fast, you can continue using CPU training:

```powershell
# Install stable CPU-only version
pip install torch torchvision torchaudio
```

Then run training without GPU flags:
```
python train_robot_rl_demo.py --steps 10000 --parallel 4
```

## Next Steps

1. We recommend trying Option 1 (DirectML) first, as it's the simplest approach for AMD GPUs on Windows
2. If you're comfortable with Linux, Option 2 (WSL2) will likely provide the best performance
3. We can modify the training script to support DirectML if you wish to go with Option 1

## Resources

- [PyTorch DirectML GitHub](https://github.com/microsoft/pytorch-directml)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [WSL2 Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install)
- [ROCm on WSL2](https://rocm.docs.amd.com/en/latest/deploy/windows.html) 