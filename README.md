# FANUC Robot ML Platform

A comprehensive machine learning platform for reinforcement learning with FANUC robot arms, focusing on precise end-effector positioning tasks. Optimized for both standard PyTorch and AMD GPU acceleration via DirectML.

## Project Overview

This platform provides a complete framework for applying reinforcement learning to control FANUC robot arms for positioning tasks. The system is designed with a modular architecture, clean Python package structure, and supports both standard PyTorch and AMD GPU acceleration via DirectML.

### Key Features

- **Reinforcement Learning**: Custom-built environments using PPO algorithm
- **Dual-GPU Support**: Works with standard PyTorch (NVIDIA) and AMD GPUs (via DirectML)
- **Flexible Evaluation Tools**: Comprehensive visualization and evaluation systems
- **Modular Design**: Well-structured Python package with clean separation of concerns
- **Unified Interface**: Single entry point for all operations

## Optimized Consolidated Codebase

This project has been streamlined to provide a unified interface for all operations:

- **Unified Entry Points**: 
  - `fanuc.bat` - Primary script for all standard operations
  - `directml.bat` - Primary script for all AMD GPU-accelerated operations
- **Single Python Implementation**: Core functionality is consolidated in `fanuc_platform.py`
- **Simplified Usage**: Consistent parameter handling across all modes
- **Reduced Complexity**: Eliminated redundant scripts and consolidated functionality

## Directory Structure

```
fanuc-ml/
├── fanuc.bat            # Primary script for all standard operations
├── directml.bat         # Primary script for all AMD GPU operations 
├── fanuc_platform.py    # Unified implementation for all operations
├── evaluate_model.bat   # Backward compatibility for model evaluation
├── test_model.bat       # Backward compatibility for quick testing
├── evaluate_directml.bat # Backward compatibility for DirectML evaluation
├── test_directml.bat    # Backward compatibility for DirectML testing
├── CONSOLIDATION.md     # Documentation of consolidation efforts
├── pyproject.toml       # Modern Python package configuration
├── requirements.txt     # Core dependencies
├── robots/              # Robot model files
│   ├── urdf/            # URDF robot descriptions
│   ├── meshes/          # 3D mesh files for robots
│   └── README.md        # Robot models documentation
├── src/                 # Source code (main package)
│   ├── __init__.py      # Package definition
│   ├── core/            # Core training and simulation code
│   ├── directml_core.py # Consolidated DirectML implementation
│   ├── envs/            # Environment implementations
│   ├── utils/           # Utility functions
│   │   ├── __init__.py  # Core utilities including seeding
│   │   └── pybullet_utils.py # PyBullet helper functions
│   └── README.md        # Source code documentation
├── tools/               # Utility tools and demos
│   └── demos/           # Robot demonstration scripts
├── models/              # Trained model storage
├── plots/               # Training plots and graphs
├── visualizations/      # Evaluation visualizations
└── docs/                # Documentation
```

## Installation

### Prerequisites

- Windows 10/11
- Python 3.8 or higher
- PyTorch for standard operation (NVIDIA GPUs)
- torch-directml for AMD GPU acceleration (optional)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fanuc-ml.git
   cd fanuc-ml
   ```

2. Basic installation:
   ```bash
   pip install -e .
   ```

3. For AMD GPU support (optional):
   ```bash
   pip install torch-directml
   ```

### Verify Installation

Run the installation test to verify your setup:

```bash
# Test standard installation
fanuc.bat install

# Test DirectML support (AMD GPUs)
directml.bat install
```

## Usage

All operations use the unified scripts with the following format:

```bash
# For standard operations (CPU or NVIDIA GPU)
fanuc.bat [mode] [options]

# For AMD GPU operations with DirectML
directml.bat [mode] [options]
```

Where `mode` can be one of:
- `train` - Train a new model or continue training an existing one
- `eval` - Run thorough evaluation on a model
- `test` - Run a quick test on a model
- `install` - Test installation

### Training a Model

```bash
# Training with default settings
fanuc.bat train

# Training with specific parameters
fanuc.bat train --model_path models/my_model --steps 1000000 --no-gui

# Training with DirectML (AMD GPUs)
directml.bat train --steps 500000
```

### Evaluating a Model

```bash
# Evaluate a model (10 episodes by default)
fanuc.bat eval models/my_model

# Evaluation with specific episode count
fanuc.bat eval models/my_model --episodes 20 --verbose

# Evaluation with DirectML
directml.bat eval models/my_model --no-gui
```

### Quick Testing

```bash
# Quick test (1 episode by default)
fanuc.bat test models/my_model

# Test with specific parameters
fanuc.bat test models/my_model --episodes 3 --verbose

# Test with DirectML
directml.bat test models/my_model
```

### Common Options

These options work with all commands:

- `--verbose` or `-v` - Enable detailed logging
- `--no-gui` - Run without visualization
- `--model_path PATH` - Specify model path (for train/eval/test)
- `--episodes N` - Number of episodes (for eval/test)
- `--steps N` - Number of training steps (for train)
- `--eval` - Run evaluation after training (for train)

## Training Parameters and Options

The training system includes several advanced parameters and options for fine-tuning model training:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--steps` | Total number of training steps | 500,000 |
| `--model_path` | Path to save/load model | Auto-generated timestamped path |
| `--eval` | Run evaluation after training | False |
| `--no-gui` | Disable GUI visualization | False |
| `--verbose` | Enable verbose output | False |
| `--seed` | Random seed for reproducibility | None |
| `--learning-rate` | Learning rate | 3e-4 |

### Training Architecture

The training implementation uses a consolidated architecture with:

1. **Reward Function**: Balanced reward function based on distance to target and action efficiency
2. **Neural Network**: Optimized feature extractor and policy network for robot control
3. **Joint Limit Handling**: Standardized enforcement of robot joint limits
4. **Memory Management**: Efficient tensor operations and resource usage

### Model Saving and Loading

Models are saved with timestamp-based naming in the `models/` directory:

```
models/fanuc-20250401-120523/    # Standard model trained on April 1, 2025
models/fanuc-20250401-120523-directml/  # DirectML model trained on April 1, 2025
```

### Monitoring and Evaluation

During training, the following metrics are tracked and saved:

- Average reward per episode
- Success rate (reaching target position)
- Average distance to target
- Average episode length
- Joint limit violations

These metrics are saved as plots in the `plots/` directory for easy monitoring.

## Advanced Usage

For more advanced usage, see the help for each mode:

```bash
fanuc.bat train --help
fanuc.bat eval --help
fanuc.bat test --help
```

## Development

### Project Structure

The project uses a modern Python package structure:

- `src/` - Main package directory containing all source code
- `src/core/` - Core functionality for robot control and training
- `src/directml_core.py` - AMD GPU-specific implementations
- `src/envs/` - Environment implementations for robot simulation
- `src/utils/` - Utility functions and tools

### Extending the Platform

To extend the platform with new features:

1. Add new modules to the appropriate directory in `src/`
2. Update the unified `fanuc_platform.py` for new functionality
3. Add docstrings for all public functions and classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PyBullet](https://pybullet.org/) - Used for physics simulation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [torch-directml](https://github.com/microsoft/DirectML) - AMD GPU acceleration

## Additional Documentation

For more detailed information, refer to:
- [Consolidation Documentation](CONSOLIDATION.md) - Details on code consolidation efforts

## Models

The repository includes pre-trained models:

- `models/fanuc-ml-directml` - DirectML trained model 

## DirectML Support for FANUC Robot Control

This project supports AMD GPU acceleration through the DirectML backend for PyTorch.

### Requirements

To use DirectML acceleration, you must have:

1. Windows 10/11 with an AMD GPU (tested with AMD RX 6700S)
2. Python 3.8+ 
3. PyTorch 2.0.0+
4. torch-directml package

### Installation

Install the DirectML backend for PyTorch:

```bash
pip install torch-directml
```

Verify the installation by running:

```bash
directml.bat install
```

### Unified DirectML Interface

The platform provides a simplified interface for DirectML-accelerated operations through the `directml.bat` script, which is a wrapper around the unified `fanuc_platform.py` implementation.

#### Training with DirectML

To train a model with AMD GPU acceleration:

```bash
directml.bat train [options]
```

Options include:
- `--model_path PATH` - Custom path to save the model
- `--steps N` - Number of training steps (default: 500000)
- `--no-gui` - Run without PyBullet visualization 
- `--eval` - Run evaluation after training
- `--verbose` - Show detailed training progress

Example:
```bash
directml.bat train --steps 1000000 --no-gui
```

#### Evaluating Models with DirectML

Evaluate a trained model using AMD GPU acceleration:

```bash
directml.bat eval <model_path> [options]
```

Options include:
- `--episodes N` - Number of evaluation episodes (default: 10)
- `--no-gui` - Run without PyBullet visualization
- `--verbose` - Show detailed episode results

Example:
```bash
directml.bat eval models/fanuc-ml-directml --episodes 20 --verbose
```

#### Quick Testing Models with DirectML

Run a quick test of a model with DirectML acceleration:

```bash
directml.bat test <model_path> [options]
```

Options include:
- `--episodes N` - Number of test episodes (default: 1)
- `--no-gui` - Run without PyBullet visualization
- `--verbose` - Show detailed results

Example:
```bash
directml.bat test models/fanuc-ml-directml --verbose
```

### Under the Hood: How DirectML Works

The DirectML implementation provides several key components:

1. **DirectML Device Detection**: The system automatically detects AMD GPUs and configures PyTorch to use them via DirectML.

2. **Optimized Model Loading**: Models trained with DirectML are loaded with DirectML-specific optimizations.

3. **Environment Variables**: The DirectML backend uses several environment variables for configuration:
   - `FANUC_DIRECTML=1` - Set by `directml.bat` to indicate DirectML mode
   - `PYTORCH_DIRECTML_VERBOSE=1` - Controls verbosity of DirectML operations
   - `DIRECTML_ENABLE_OPTIMIZATION=1` - Enables DirectML optimization (default)

### High-Level Architecture

```
directml.bat → fanuc_platform.py → src/directml_core.py
```

- `directml.bat`: Entry point script that adds the `--directml` flag
- `fanuc_platform.py`: Unified platform script that handles all operations
- `src/directml_core.py`: Core DirectML implementation with AMD-specific optimizations

### Advanced Configuration

DirectML behavior can be fine-tuned with environment variables:

- `PYTORCH_DIRECTML_VERBOSE`: Set to 1 for verbose output (default: 0)
- `DIRECTML_ENABLE_OPTIMIZATION`: Enable DirectML optimization (default: 1)
- `DIRECTML_GPU_TRANSFER_BIT_WIDTH`: Bit width for GPU transfers (default: 64)

### Troubleshooting

Common issues with DirectML:

1. **No GPU detected**: Make sure your AMD drivers are up to date.
2. **Out of memory errors**: Reduce batch size or model complexity.
3. **Performance issues**: Check for background processes using GPU resources.

### Development Notes

When developing new features that need to support DirectML:

1. Use the `torch_directml.device()` for tensor operations instead of hardcoding device references
2. Use the utility function `is_available()` from `src.directml_core` to check for DirectML availability
3. Handle DirectML errors with appropriate error messages rather than fallbacks
4. Add your implementation to the `src/directml_core.py` module for consistency 