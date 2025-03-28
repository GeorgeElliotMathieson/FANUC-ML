# FANUC-ML Tools

This directory contains standalone utilities and tools for the FANUC Robot ML Platform that are not part of the core package but provide additional functionality.

## Directory Structure

- `demos/`: Demonstration scripts for the FANUC robot
  - `load_fanuc_robot.py`: Basic script to load and control a FANUC robot in PyBullet
  
- `run_eval.py`: Specialized tool for evaluating DirectML models
- `test_install.py`: Utility to verify the correct installation of the package

## Using the Tools

### Evaluation Tool

The `run_eval.py` script provides a specialized evaluation tool for DirectML models that avoids argument parsing conflicts:

```bash
# Evaluate a DirectML model
python tools/run_eval.py ./models/my_directml_model 5 --verbose
```

Arguments:
1. Model path (required)
2. Number of episodes (optional, default: 5)
3. Options: `--verbose`, `--no-gui`, `--speed=X`

### Installation Tester

The `test_install.py` script verifies that all required components of the FANUC-ML package are correctly installed:

```bash
# Test installation
python tools/test_install.py
```

### Demo Scripts

The demos directory contains standalone demonstration scripts:

```bash
# Run the FANUC robot demo
python tools/demos/load_fanuc_robot.py
```

## Development Guidelines

When adding new tools:

1. Place standalone utilities in the root of the tools directory
2. Add demonstration scripts to the demos subdirectory
3. Include comprehensive documentation in each script
4. Make sure scripts can run independently with minimal dependencies
5. Add usage information to this README 