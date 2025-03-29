# FANUC Robot ML Platform (DirectML Edition)

A comprehensive machine learning platform for reinforcement learning with FANUC robot arms, focusing on precise end-effector positioning tasks. Optimized exclusively for AMD GPU acceleration via DirectML.

## Project Overview

This platform provides a complete framework for applying reinforcement learning to control FANUC robot arms for positioning tasks. The system is designed with a modular architecture, clean Python package structure, and is specifically optimized for AMD GPU acceleration via DirectML.

### Key Features

- **Reinforcement Learning**: Custom-built environments using PPO algorithm
- **AMD GPU Support**: Exclusively works with AMD GPUs (via DirectML)
- **Flexible Evaluation Tools**: Comprehensive visualization and evaluation systems
- **Modular Design**: Well-structured Python package with clean separation of concerns
- **Unified Interface**: Single entry point for all operations

## Streamlined Architecture

This project has been streamlined to provide a single unified interface for all operations:

- **Single Entry Point**: 
  - `fanuc.bat` - The only script needed for all operations (train, evaluate, test, install)
- **Consolidated Implementation**: Core functionality in `fanuc_platform.py` and `dml.py`
- **Simplified Usage**: Consistent parameter handling across all modes

## Directory Structure

```
fanuc-ml/
├── fanuc.bat            # Single entry point for all operations
├── fanuc_platform.py    # Unified implementation for all operations
├── README.md            # This comprehensive documentation
├── pyproject.toml       # Modern Python package configuration
├── requirements.txt     # Core dependencies
├── robots/              # Robot model files
│   ├── urdf/            # URDF robot descriptions
│   ├── meshes/          # 3D mesh files for robots
├── src/                 # Source code (main package)
│   ├── __init__.py      # Package definition
│   ├── core/            # Core training and simulation code
│   │   ├── __init__.py  # Core module definition
│   │   └── robot_trainer.py  # Main training implementation
│   ├── dml.py           # DirectML implementation
│   ├── envs/            # Environment implementations
│   ├── utils/           # Utility functions
│   │   ├── __init__.py  # Core utilities including seeding
│   │   └── pybullet_utils.py # PyBullet helper functions
├── tools/               # Utility tools and demos
│   └── demos/           # Robot demonstration scripts
├── models/              # Trained model storage
├── plots/               # Training plots and graphs
└── visualizations/      # Evaluation visualizations
```

## Installation

### Prerequisites

- Windows 10/11
- Python 3.8 or higher
- AMD GPU
- torch-directml package

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fanuc-ml.git
   cd fanuc-ml
   ```

2. Installation:
   ```bash
   pip install -e .
   pip install torch-directml
   ```

### Verify Installation

Run the installation test to verify your setup:

```bash
# Test DirectML installation
fanuc.bat install
```

## Usage

All operations use the single unified script with the following format:

```bash
# For all operations (AMD GPU with DirectML)
fanuc.bat [mode] [options]

# If using PowerShell, use:
.\fanuc.bat [mode] [options]

# Alternatively, you can run directly with Python:
python fanuc_platform.py [mode] [options]
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
```

### Evaluating a Model

```bash
# Evaluate a model (10 episodes by default)
fanuc.bat eval models/my_model

# Evaluation with specific episode count
fanuc.bat eval models/my_model --episodes 20 --verbose
```

### Quick Testing

```bash
# Quick test (1 episode by default)
fanuc.bat test models/my_model

# Test with specific parameters
fanuc.bat test models/my_model --episodes 3 --verbose
```

### Demo Scripts

The platform includes demo scripts for exploring robot functionality:

```bash
# Run the FANUC robot demo
python tools/demos/load_fanuc_robot.py
```

#### Available Demos

- `load_fanuc_robot.py` - Basic script to load a FANUC robot in PyBullet and demonstrate joint motion with a sequence of predefined poses.

#### Demo Features

- Robot loading and visualization
- Joint control and movement
- Simulation parameter optimization
- Examples of how to interact with the PyBullet API

These demos serve as practical examples for developers who want to understand the low-level robot control without diving into the full reinforcement learning framework.

### Common Options

These options work with all commands:

- `--verbose` or `-v` - Enable detailed logging
- `--no-gui` - Run without visualization
- `--model_path PATH` - Specify model path (for train/eval/test)
- `--episodes N` - Number of episodes (for eval/test)
- `--steps N` - Number of training steps (for train)
- `--eval` - Run evaluation after training (for train)

### Troubleshooting

#### Command Execution Issues

- **PowerShell Users**: When using PowerShell, prefix batch files with `.\` (e.g., `.\fanuc.bat install`)
- **Command Not Found**: If you encounter issues with the batch file, you can run commands directly with Python: `python fanuc_platform.py [command] [options]`
- **Permission Issues**: On some systems, you may need to adjust execution policies for running batch files. Try running PowerShell as administrator.

#### DirectML Issues

Common issues with DirectML:

1. **No GPU detected**: Make sure your AMD drivers are up to date.
2. **Out of memory errors**: Reduce batch size or model complexity.
3. **Performance issues**: Check for background processes using GPU resources.

## Robot Model Specifications

The FANUC robot models have:
- 6 revolute joints for the arm
- Accurate inertial properties from the original FANUC specifications
- Realistic joint limits and constraints
- Properly configured collision geometries for simulation

### Using Robot Models in Your Projects

To use these robot models in your own PyBullet projects:

```python
import pybullet as p
import os

# Initialize PyBullet
p.connect(p.GUI)

# Get the path to the URDF file - update path as needed
urdf_path = os.path.join("robots", "urdf", "fanuc.urdf")
robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# Control the robot
# ...
```

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
models/fanuc-20250401-120523/  # Model trained on April 1, 2025
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

### Module Structure

The codebase is organized into logical modules:

- `core/`: Core reinforcement learning algorithms and training logic
  - `robot_trainer.py`: Main implementation of the PPO-based robot positioning task

- `dml.py`: DirectML support for AMD GPU acceleration
  - Contains DirectML-specific implementations and utilities
  - Implements DirectML-optimized model loading and inference

- `envs/`: Robot environment implementations
  - Contains PyBullet-based robot simulation environments

- `utils/`: Utility functions and helpers
  - Contains PyBullet helpers and general utilities

### Module Dependencies

The modules are designed with the following dependency flow:

```
fanuc_platform.py → src/dml.py → src/core/robot_trainer.py
                              → src/envs/robot_sim.py
                            
src/utils/* ← used by all modules
```

### Extending the Platform

To extend the platform with new features:

1. Add new modules to the appropriate directory in `src/`
2. Update the unified `fanuc_platform.py` for new functionality
3. Add docstrings for all public functions and classes

### Development Guidelines

When adding new tools:

1. Add new functionality to the consolidated `fanuc_platform.py` file instead of creating separate scripts
2. Add demonstration scripts to the tools/demos subdirectory
3. Include comprehensive documentation in each script
4. Maintain consistent naming schemes and coding standards
5. Follow the established modular architecture

### Naming Conventions

- Module names: lowercase with underscores (e.g., `robot_sim.py`)
- Class names: CamelCase (e.g., `FANUCRobotEnv`)
- Functions and methods: lowercase with underscores (e.g., `train_robot()`)
- Constants: UPPERCASE with underscores (e.g., `MAX_STEPS`)

## DirectML Implementation Details

This platform exclusively uses AMD GPU acceleration through the DirectML backend for PyTorch.

### Requirements

To use this platform, you must have:

1. Windows 10/11 with an AMD GPU (tested with AMD RX 6700S)
2. Python 3.8+ 
3. PyTorch 2.0.0+
4. torch-directml package

### DirectML Architecture

The DirectML implementation provides several key components:

1. **DirectML Device Detection**: The system automatically detects AMD GPUs and configures PyTorch to use them via DirectML.

2. **Optimized Model Loading**: Models are loaded with DirectML-specific optimizations.

3. **Environment Variables**: The DirectML backend uses several environment variables for configuration:
   - `FANUC_DIRECTML=1` - Set by `fanuc.bat` to indicate DirectML mode
   - `PYTORCH_DIRECTML_VERBOSE=1` - Controls verbosity of DirectML operations
   - `DIRECTML_ENABLE_OPTIMIZATION=1` - Enables DirectML optimization (default)

### Advanced Configuration

DirectML behavior can be fine-tuned with environment variables:

- `PYTORCH_DIRECTML_VERBOSE`: Set to 1 for verbose output (default: 0)
- `DIRECTML_ENABLE_OPTIMIZATION`: Enable DirectML optimization (default: 1)
- `DIRECTML_GPU_TRANSFER_BIT_WIDTH`: Bit width for GPU transfers (default: 64)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PyBullet](https://pybullet.org/) - Used for physics simulation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [torch-directml](https://github.com/microsoft/DirectML) - AMD GPU acceleration
- [sezan92/Fanuc](https://github.com/sezan92/Fanuc.git) - Original robot model base

## Pre-trained Models

The repository includes pre-trained models:

- `models/fanuc-ml-amd` - DirectML trained model 
