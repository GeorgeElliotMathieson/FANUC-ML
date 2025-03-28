# DirectML Support for FANUC Robot Control

This project **requires** AMD GPU acceleration through the DirectML backend for PyTorch. This document explains how to use DirectML features and optimize your workflow for AMD GPUs.

## Requirements

To run this project, you **must** have:

1. Windows 10/11 with an AMD GPU (tested with AMD RX 6700S)
2. Python 3.8+ 
3. PyTorch 2.0.0+
4. torch-directml package

## Installation

Install the DirectML backend for PyTorch:

```bash
pip install torch-directml
```

Verify the installation by running:

```bash
python tools/test_install.py
```

## Using DirectML Features

### Training with DirectML

To train a model:

```bash
python main.py --train
```

This will automatically detect your AMD GPU and configure the training process to use DirectML.

### Evaluating DirectML Models

You can evaluate a trained model using:

```bash
scripts/directml/test_directml.bat
```

Or for more options:

```bash
python scripts/directml/test_directml_model.py --model models/your_model --episodes 5
```

### Advanced Configuration

DirectML behavior can be tuned with environment variables:

- `PYTORCH_DIRECTML_VERBOSE`: Set to 1 for verbose output (default: 0)
- `DIRECTML_ENABLE_OPTIMIZATION`: Enable DirectML optimization (default: 1)
- `DIRECTML_GPU_TRANSFER_BIT_WIDTH`: Bit width for GPU transfers (default: 64)

## Troubleshooting

Common issues with DirectML:

1. **No GPU detected**: Make sure your AMD drivers are up to date.
2. **Out of memory errors**: Reduce batch size or model complexity.
3. **Performance issues**: Check for background processes using GPU resources.

## Development Notes

When developing new features:

1. Use the `torch_directml.device()` for tensor operations instead of hardcoding device references
2. Use the utility function `is_directml_available()` to check for DirectML availability
3. Handle DirectML errors with appropriate error messages rather than fallbacks

## Reference

For more information on DirectML and PyTorch, see:
- [torch-directml Documentation](https://github.com/microsoft/DirectML)
- [AMD ROCm Documentation](https://www.amd.com/en/graphics/servers-solutions-rocm) 