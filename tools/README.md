# FANUC-ML Tools

This directory contains standalone utilities and tools for the FANUC Robot ML Platform that are not part of the core package but provide additional functionality.

## Directory Structure

- `demos/`: Demonstration scripts for the FANUC robot
  - `load_fanuc_robot.py`: Basic script to load and control a FANUC robot in PyBullet
  
- `fanuc_tools.py`: Consolidated toolset for various utilities:
  - Installation testing (install)
  - Model evaluation (eval)
  - Model quick testing (test)

## Using the Tools

### Consolidated Tools Interface

The `fanuc_tools.py` script provides a unified interface to multiple tools:

```bash
# Test installation
python tools/fanuc_tools.py install

# Evaluate a model thoroughly
python tools/fanuc_tools.py eval ./models/my_model 5 --verbose

# Run a quick test of a model
python tools/fanuc_tools.py test ./models/my_model --speed=0.05
```

### Model Testing and Evaluation

The tool supports two modes for assessing models:

1. **Quick Test** (`test`): Runs a minimal test with fewer episodes for quick verification
   ```bash
   python tools/fanuc_tools.py test ./models/my_model
   ```
   
2. **Thorough Evaluation** (`eval`): Runs a more comprehensive evaluation with more episodes
   ```bash
   python tools/fanuc_tools.py eval ./models/my_model 10 --verbose
   ```

Arguments for both modes:
1. Model path (required)
2. Number of episodes (optional, default: 1 for test, 5 for eval)
3. Options: `--verbose`, `--no-gui`, `--speed=X`

### Installation Tester

The installation tester verifies that all required components of the FANUC-ML package are correctly installed:

```bash
# Test installation
python tools/fanuc_tools.py install
```

### Demo Scripts

The demos directory contains standalone demonstration scripts:

```bash
# Run the FANUC robot demo
python tools/demos/load_fanuc_robot.py
```

## Batch Scripts

For convenience, several batch scripts are provided for easy access:

- Root directory scripts:
  - `directml.bat`: Unified script for all DirectML operations (train, eval, test, install)
  - `evaluate_model.bat`: Evaluates a model thoroughly
  - `test_model.bat`: Runs a quick test of a model
  - `evaluate_directml.bat`: Backward compatibility for DirectML evaluation
  - `test_directml.bat`: Backward compatibility for DirectML testing

### DirectML Integration

The tools are designed to seamlessly work with DirectML-accelerated models:

- When using `directml.bat`, the environment variable `USE_DIRECTML=1` is automatically set
- When running evaluation, DirectML-specific model loading is used when appropriate
- The tools automatically detect if a model was trained with DirectML

## Development Guidelines

When adding new tools:

1. Add new functionality to the consolidated `fanuc_tools.py` file instead of creating separate scripts
2. Add demonstration scripts to the demos subdirectory
3. Include comprehensive documentation in each script
4. Make sure scripts can run independently with minimal dependencies
5. Add usage information to this README 