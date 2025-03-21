@echo off
echo Setting up environment for AMD GPU training...

:: Set environment variables for AMD GPU
set HIP_VISIBLE_DEVICES=0
set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

:: Run the training script with AMD GPU support
echo Starting training with AMD GPU acceleration...
python train_robot_rl_demo.py --use-amd --steps 10000 --parallel 8

echo.
echo Training complete.
echo Press any key to exit...
pause > nul 