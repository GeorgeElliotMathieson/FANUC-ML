# DirectML Integration for AMD GPUs

This folder contains a modified version of the robot training script that supports AMD GPUs using DirectML. The script `train_robot_rl_demo_directml.py` is designed to work with AMD Radeon GPUs on Windows by using the PyTorch DirectML backend.

## Installation

1. Make sure you have installed the DirectML backend for PyTorch:
   ```
   pip install torch-directml
   ```

2. Set the following environment variables to optimize AMD GPU performance:
   ```
   set HIP_VISIBLE_DEVICES=0
   set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
   ```

   You can also set these permanently using the `set_permanent_amd_vars.ps1` PowerShell script (run as Administrator).

## Usage

### Training

To train a model using your AMD GPU with DirectML acceleration:

```
python train_robot_rl_demo_directml.py --steps 30000 --parallel 4
```

Key options:
- `--steps`: Number of training steps (default: 300000)
- `--parallel`: Number of parallel environments (default: 8)
- `--memory-limit`: Memory limit for DirectML in MB (default: 128)
- `--disable-directml`: Force CPU usage even if DirectML is available

### Evaluation

To evaluate a trained model:

```
python train_robot_rl_demo_directml.py --eval-only --load ./models/revamped_XXXXXXXX_XXXXXX/final_model --eval-episodes 10
```

### Demo

To run a demo with a trained model:

```
python train_robot_rl_demo_directml.py --demo --load ./models/revamped_XXXXXXXX_XXXXXX/final_model --viz-speed 0.05
```

## How It Works

The script uses a technique called "monkey patching" to redirect PyTorch's CUDA calls to DirectML. This allows standard PyTorch code that would normally use NVIDIA GPUs to work with AMD GPUs. Key components include:

1. **Patching PyTorch Functions**: The script overrides `torch.device`, `torch.Tensor.to`, and other CUDA-related functions to use DirectML instead.

2. **Synchronization Handling**: AMD GPUs with DirectML require explicit synchronization, which the script handles through helper functions.

3. **Model Wrapping**: During inference, models are wrapped with a DirectML-aware wrapper that ensures tensors are properly moved to and from the GPU.

## Limitations

- DirectML support on Windows is still evolving and may not provide the same performance as CUDA on NVIDIA GPUs.
- Some operations may fall back to CPU if not supported by DirectML.
- Memory usage needs to be carefully managed with the `--memory-limit` option.
- The script may require updates when using with newer versions of PyTorch or DirectML.

## Troubleshooting

If you encounter issues with DirectML acceleration:

1. Ensure you have the latest GPU drivers installed.
2. Try running with `--memory-limit 64` to reduce GPU memory usage.
3. If training is unstable, try reducing the number of parallel environments with `--parallel 1`.
4. Check that the `torch-directml` package is installed correctly.
5. Verify that the DirectML device is detected by running the diagnostic script:
   ```
   python test_amd_directml_performance.py
   ```

## Quick Test

You can use the provided batch file to test DirectML acceleration:

```
test_amd_directml_training.bat
```

This will run a short training session and provide guidance for further use. 