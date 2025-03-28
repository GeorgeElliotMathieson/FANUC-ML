# FANUC-ML Source Code

This directory contains the core Python modules for the FANUC Robot ML Platform. The codebase is organized into logical subdirectories according to functionality.

## Directory Structure

- `core/`: Core reinforcement learning algorithms and training logic
  - `train_robot_rl_positioning_revamped.py`: Main implementation of the PPO-based robot positioning task

- `directml_core.py`: Consolidated DirectML support for AMD GPU acceleration
  - Contains DirectML-specific implementations and utilities
  - Provides functions for checking DirectML availability, device setup
  - Implements DirectML-optimized model loading and inference
  - Contains the complete implementation of DirectML PPO and evaluation/testing functions

- `envs/`: Robot environment implementations
  - `robot_sim.py`: PyBullet-based robot simulation environment

- `utils/`: Utility functions and helpers
  - `pybullet_utils.py`: PyBullet helper functions
  - Other general utilities shared across modules

## Module Dependencies

The modules are designed with the following dependency flow:

```
fanuc_platform.py → src/directml_core.py → src/core/train_robot_rl_positioning_revamped.py
                                        → src/envs/robot_sim.py
                            
src/utils/* ← used by all modules
```

## Usage

Most users will not need to interact with these modules directly. Instead, use the unified entry points:

```bash
# For standard operations (CPU or NVIDIA GPU)
fanuc.bat [mode] [options]

# For AMD GPU operations with DirectML
directml.bat [mode] [options]
```

## Development Notes

When extending the codebase:

1. Keep core algorithms in the `core/` directory
2. Place GPU-specific code in `directml_core.py`
3. Add new robot environments to `envs/`
4. Place shared utilities in `utils/`
5. Update imports in `__init__.py` files for components that should be exposed
6. Maintain unified interfaces in `fanuc_platform.py`

## Naming Conventions

- Module names: lowercase with underscores (e.g., `robot_sim.py`)
- Class names: CamelCase (e.g., `FANUCRobotEnv`)
- Functions and methods: lowercase with underscores (e.g., `train_revamped_robot()`)
- Constants: UPPERCASE with underscores (e.g., `MAX_STEPS`) 