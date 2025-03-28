# DirectML Scripts for FANUC Robot Control

This directory contains specialized scripts for working with DirectML-accelerated models for the FANUC robot positioning task.

## Available Scripts

### `test_directml.bat`

A Windows batch script for testing DirectML models with a simple, visual interface.

```batch
# Default usage (uses the default DirectML model with 1 episode)
test_directml.bat

# With custom model and episode count
test_directml.bat path/to/model 3
```

### `test_directml_model.py`

A Python script for testing DirectML models with detailed metrics and visualization options.

```bash
# Basic usage
python test_directml_model.py

# All options
python test_directml_model.py --model PATH --episodes N --no-gui --viz-speed 0.02
```

### `evaluate_directml.bat`

A Windows batch script for evaluating DirectML models with more configuration options.

```batch
# Default usage
evaluate_directml.bat

# With custom parameters
evaluate_directml.bat path/to/model 5 "--no-gui --verbose"
```

## Root Directory Convenience Scripts

For ease of use, the following convenience scripts are provided in the project root directory:

- `test_directml.bat`: Forwards to `scripts/directml/test_directml.bat`
- `evaluate_directml.bat`: Forwards to `scripts/directml/evaluate_directml.bat`

These scripts can be called from the project root with the same parameters as their counterparts in this directory.

## DirectML Support

These scripts are designed to work with AMD GPUs through the DirectML interface. They provide:

1. Proper device handling for DirectML models
2. Compatible model loading and evaluation
3. Visual demonstration of model performance
4. Detailed metrics on model accuracy

For more information on DirectML support, see `README_DIRECTML.md` in the project root. 