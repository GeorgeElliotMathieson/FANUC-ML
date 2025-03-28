# FANUC-ML Source Code

This directory contains the core Python modules for the FANUC Robot ML Platform. The codebase is organized into logical subdirectories according to functionality.

## Directory Structure

- `core/`: Core reinforcement learning algorithms and training logic
  - `train_robot_rl_positioning_revamped.py`: Main implementation of the PPO-based robot positioning task

- `directml/`: AMD GPU acceleration support via DirectML
  - `directml_train.py`: Entry point for DirectML-accelerated training
  - `train_robot_rl_ppo_directml.py`: DirectML-specific PPO implementation

- `envs/`: Robot environment implementations
  - `robot_sim.py`: PyBullet-based robot simulation environment

- `utils/`: Utility functions and helpers
  - `pybullet_utils.py`: PyBullet helper functions
  - `seed.py`: Random seed utilities

- `train_robot.py`: Main training script and entry point

## Module Dependencies

The modules are designed with the following dependency flow:

```
main.py → src/train_robot.py → src/core/train_robot_rl_positioning_revamped.py
                            → src/directml/directml_train.py
                            
src/envs/robot_sim.py ← used by both core and directml modules
src/utils/* ← used by all modules
```

## Usage

Most users will not need to interact with these modules directly. Instead, use the main entry point:

```bash
# Using the main entry point
python main.py --train
```

## Development Notes

When extending the codebase:

1. Keep core algorithms in the `core/` directory
2. Place GPU-specific code in `directml/`
3. Add new robot environments to `envs/`
4. Place shared utilities in `utils/`
5. Update imports in `__init__.py` files for components that should be exposed
6. Maintain backward compatibility with existing scripts

## Naming Conventions

- Module names: lowercase with underscores (e.g., `robot_sim.py`)
- Class names: CamelCase (e.g., `FANUCRobotEnv`)
- Functions and methods: lowercase with underscores (e.g., `train_revamped_robot()`)
- Constants: UPPERCASE with underscores (e.g., `MAX_STEPS`) 