# Revamped Robot Positioning with Reinforcement Learning

This implementation provides a completely revamped approach to training a robotic arm for precise end-effector positioning using reinforcement learning. The new approach addresses several shortcomings in the original implementation that led to suboptimal performance and undesirable convergence.

## Key Improvements

1. **Action Space Representation**:
   - Changed from velocity control to delta joint position control
   - Allows for more precise and predictable movements
   - Reduces the likelihood of jerky or oscillating behaviors

2. **Sophisticated Reward Engineering**:
   - Multi-component reward structure with tuned weights:
     - Distance component: exponential function providing higher gradients closer to target
     - Progress component: rewards improvements in position relative to previous state
     - Efficiency penalty: discourages excessive movement
     - Smoothness penalty: promotes continuous, smooth motion
     - Joint limit penalty: encourages staying away from joint limits

3. **Enhanced Neural Network Architecture**:
   - Structured feature extraction with component-specific encoders
   - Residual connections for better gradient flow
   - Layer normalization for improved training stability
   - Separation of different observation components (joint state, position, target, history)

4. **Improved Observation Space**:
   - Includes temporal information with history of past states
   - Better feature normalization and encoding
   - Direct relative vectors to target position

5. **Curriculum Learning**:
   - Gradually increases difficulty as the agent improves
   - Starts with closer targets and expands the workspace
   - Adjusts sampling strategies based on agent performance

6. **Target Position Sampling Strategies**:
   - Multiple strategies for comprehensive coverage of the workspace
   - Uniform sampling, outer region sampling, precision region sampling, vertical stacking

7. **Training Enhancements**:
   - Observation and reward normalization
   - Shorter episodes for more efficient learning
   - Improved logging and monitoring
   - Support for parallel training environments

## Usage

### Training a New Model

```bash
python train_robot_rl_positioning_revamped.py --parallel 8 --steps 1000000 --learning-rate 3e-4
```

### Continuing Training from a Checkpoint

```bash
python train_robot_rl_positioning_revamped.py --load ./models/revamped_xyz/final_model --steps 500000
```

### Evaluating a Trained Model

```bash
python train_robot_rl_positioning_revamped.py --eval-only --load ./models/revamped_xyz/final_model --eval-episodes 20
```

### Running a Demo with a Trained Model

```bash
python train_robot_rl_positioning_revamped.py --demo --load ./models/revamped_xyz/final_model --viz-speed 0.05
```

### Creating a Video of the Trained Agent

```bash
python train_robot_rl_positioning_revamped.py --demo --load ./models/revamped_xyz/final_model --save-video
```

## Command-Line Arguments

- `--steps`: Total number of training steps (default: 1,000,000)
- `--load`: Path to a pre-trained model to continue training or for evaluation
- `--eval-only`: Only run evaluation on a pre-trained model
- `--eval-episodes`: Number of episodes for evaluation (default: 10)
- `--demo`: Run a demonstration sequence with the model
- `--save-video`: Save a video of the evaluation
- `--gui`: Enable GUI visualization (default: true)
- `--no-gui`: Disable GUI visualization
- `--parallel`: Number of parallel environments (default: 8)
- `--parallel-viz`: Enable parallel visualization (multiple robots in single environment)
- `--viz-speed`: Control visualization speed (delay in seconds)
- `--learning-rate`: Learning rate for the optimizer (default: 3e-4)
- `--use-cuda`: Use CUDA for training if available (default: true)
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable verbose output

## Technical Details

### Environment

The revamped environment (`RobotPositioningRevampedEnv`) provides:

1. **Action Space**: Continuous space representing delta joint positions
2. **Observation Space**: Comprehensive state representation including:
   - Current normalized joint angles
   - Current end-effector position
   - Target position
   - Distance to target
   - Normalized direction to target
   - Previous action
   - History of joint positions and end-effector positions

### Neural Network

The custom neural architecture includes:

1. **Feature Extractor**: Processes different components of the observation separately
   - Joint encoder for processing joint positions and previous actions
   - Position encoder for end-effector, target positions, and direction vectors
   - Distance encoder for scalar distance values
   - History encoder for temporal information

2. **Actor Network**: Produces actions with residual connections
   - Multiple fully-connected layers with layer normalization
   - Residual connections for better gradient flow
   - Tanh activation for final output

3. **Critic Network**: Estimates value function
   - Similar architecture to actor network
   - Outputs scalar value estimates

### Reinforcement Learning Algorithm

The implementation uses Proximal Policy Optimization (PPO) with:
- Normalized observations and rewards
- Multiple parallel environments
- Adaptive learning rate
- Entropy coefficient for exploration

## Results and Performance

The revamped approach significantly outperforms the original implementation:
- Higher success rate in reaching targets
- More consistent behavior across the workspace
- Smoother and more efficient movements
- Better generalization to unseen target positions

## Development and Extension

This implementation is designed to be easily extended and modified. Key areas for potential improvement:

1. Fine-tuning the reward components and their weights
2. Exploring different network architectures
3. Adding obstacle avoidance capabilities
4. Implementing multi-goal training for improved generalization 