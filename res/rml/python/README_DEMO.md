# Robot Positioning Demo Wrapper

This wrapper script (`train_robot_rl_demo.py`) provides a simplified interface for training, evaluating, and demonstrating robot positioning models. It handles different API versions and makes it easy to use the robot positioning functionality.

## Features

- **Demo Mode**: Visualize a pre-trained model in action
- **Evaluation Mode**: Systematically evaluate a model's performance
- **Training Mode**: Train a new model or continue training an existing one
- **API Compatibility**: Works with different versions of Gymnasium/Gym
- **Video Recording**: Option to save demo videos
- **Joint Limit Enforcement**: Strictly enforces joint angle limits from the URDF model

## Usage Examples

### Training a Model

Train a new model for a specified number of steps:

```
python train_robot_rl_demo.py --steps 10000 --parallel 4
```

Continue training an existing model:

```
python train_robot_rl_demo.py --steps 5000 --load ./models/YOUR_MODEL_PATH/final_model
```

### Demonstrating a Pre-trained Model

Run a demonstration with a pre-trained model:

```
python train_robot_rl_demo.py --demo --load ./models/YOUR_MODEL_PATH/final_model
```

Adjust visualization speed (higher value = slower):

```
python train_robot_rl_demo.py --demo --load ./models/YOUR_MODEL_PATH/final_model --viz-speed 0.05
```

### Evaluating a Pre-trained Model

Evaluate a model for a specific number of episodes:

```
python train_robot_rl_demo.py --eval-only --load ./models/YOUR_MODEL_PATH/final_model --eval-episodes 5
```

### Recording a Demo Video

Record a video of the demonstration:

```
python train_robot_rl_demo.py --demo --load ./models/YOUR_MODEL_PATH/final_model --save-video
```

## Command-line Arguments

- `--steps INT`: Number of training steps (default: 10000)
- `--load PATH`: Path to a pre-trained model to load
- `--eval-only`: Run evaluation only
- `--eval-episodes INT`: Number of episodes for evaluation (default: 5)
- `--demo`: Run in demonstration mode
- `--save-video`: Save a video of the demonstration
- `--viz-speed FLOAT`: Visualization speed/delay in seconds (default: 0.02)
- `--parallel INT`: Number of parallel environments for training (default: 1)
- `--gui`: Enable GUI visualization (default: True)
- `--no-gui`: Disable GUI visualization
- `--learning-rate FLOAT`: Learning rate for the optimizer (default: 3e-4)
- `--seed INT`: Random seed for reproducibility
- `--verbose`: Enable verbose output (default: True)
- `--strict-limits`: Strictly enforce joint limits from URDF model (default: True)

## Joint Limit Enforcement

This wrapper implements strict enforcement of joint limits as defined in the URDF model. This ensures that:

1. The robot joints never move beyond their physical limits
2. Actions that would violate joint limits are automatically adjusted
3. Any model's actions are constrained to physically feasible movements

When a joint limit violation is detected, a warning message is displayed showing the attempted position and the enforced limit.

## Troubleshooting

If you encounter "unrecognized arguments" errors with `--demo` or other flags, make sure you're using this wrapper script (`train_robot_rl_demo.py`) and not the original implementation.

