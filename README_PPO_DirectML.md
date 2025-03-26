# FANUC Robot Positioning with PPO and DirectML

This implementation trains a FANUC robot arm to position its end-effector at target locations using Proximal Policy Optimization (PPO) algorithm with DirectML support specifically for AMD RX 6700S GPU.

## Features

- **PPO Algorithm**: More stable policy optimization compared to SAC
- **AMD GPU Acceleration**: Uses DirectML for PyTorch to run on AMD RX 6700S GPU
- **Visualization Support**: Real-time visualization of training progress
- **Performance Metrics**: Comprehensive tracking and plotting of training metrics
- **Lean Implementation**: Minimal code footprint for better maintainability

## Requirements

- Python 3.8+
- AMD RX 6700S GPU (or other compatible AMD GPU)
- PyTorch 1.13.0+
- torch-directml

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the robot using PPO with DirectML on your AMD GPU:

```bash
python train_robot_rl_ppo_directml.py --steps 1000000 --parallel 8
```

### Options

- `--steps`: Total number of timesteps for training (default: 1000000)
- `--parallel`: Number of parallel environments (default: 8)
- `--viz-speed`: Visualization speed, higher values slow down visualization (default: 0.0)
- `--eval-freq`: Frequency of evaluation during training (default: 10000)
- `--save-freq`: Frequency of model saving (default: 50000)
- `--no-gpu`: Use CPU instead of GPU
- `--load`: Path to load a pre-trained model
- `--eval-only`: Only run evaluation on a pre-trained model
- `--verbose`: Enable verbose output
- `--seed`: Random seed for reproducibility

### PPO Specific Parameters

- `--learning-rate`: Learning rate (default: 3e-4)
- `--n-steps`: Number of steps per PPO update (default: 2048)
- `--batch-size`: Minibatch size for updates (default: 64)
- `--n-epochs`: Number of optimization epochs per update (default: 10)
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda parameter (default: 0.95)
- `--clip-range`: PPO clip range (default: 0.2)
- `--ent-coef`: Entropy coefficient (default: 0.0)
- `--vf-coef`: Value function coefficient (default: 0.5)
- `--max-grad-norm`: Maximum gradient norm (default: 0.5)

## Examples

### Basic Training

```bash
python train_robot_rl_ppo_directml.py
```

### Training with Visualization

```bash
python train_robot_rl_ppo_directml.py --viz-speed 0.01
```

### Evaluation Only

```bash
python train_robot_rl_ppo_directml.py --eval-only --load ./models/ppo_directml_20230515_123456/final_model.pt
```

### Training with Custom PPO Parameters

```bash
python train_robot_rl_ppo_directml.py --learning-rate 2e-4 --clip-range 0.1 --n-epochs 15
```

## Outputs

- Trained models saved in `./models/{run_name}/`
- Training logs stored in `./logs/{run_name}/`
- Performance plots saved in `./plots/{run_name}/`

## Notes on DirectML for AMD GPUs

DirectML provides hardware acceleration for deep learning on AMD GPUs. This implementation automatically configures PyTorch to use DirectML when an AMD GPU is detected. If DirectML cannot be initialized, the system will fall back to CPU processing.

To verify DirectML is working properly, check for the "DirectML setup successful!" message in the console output at the start of training.

## Comparison with SAC Algorithm

PPO (used in this implementation) offers several advantages over the SAC algorithm:

1. **Stability**: Generally more stable during training
2. **Sample Efficiency**: Often requires fewer samples to achieve good performance
3. **Hyperparameter Sensitivity**: Less sensitive to hyperparameter tuning
4. **Memory Usage**: Typically uses less memory than SAC

## Performance Tips

- Increase `--parallel` based on your CPU core count for faster training
- Adjust `--batch-size` based on your GPU memory
- Use `--viz-speed 0.0` during training for maximum performance (no visualization)
- Enable visualization with higher `--viz-speed` values when evaluating models

## License

This project is licensed under the MIT License - see the LICENSE file for details. 