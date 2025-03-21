@echo off
echo Testing DirectML-accelerated training script

REM Set environment variables for AMD GPU
set HIP_VISIBLE_DEVICES=0
set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

echo.
echo Using environment variables:
echo HIP_VISIBLE_DEVICES=%HIP_VISIBLE_DEVICES%
echo PYTORCH_HIP_ALLOC_CONF=%PYTORCH_HIP_ALLOC_CONF%
echo.

echo Running a short training session with DirectML acceleration...
python train_robot_rl_demo_directml.py --steps 1000 --parallel 1

echo.
echo To run a longer training session, try:
echo python train_robot_rl_demo_directml.py --steps 30000 --parallel 4
echo.

echo To run evaluation with the trained model:
echo python train_robot_rl_demo_directml.py --eval-only --load ./models/revamped_XXXXXXXX_XXXXXX/final_model
echo.

echo To run demo with the trained model:
echo python train_robot_rl_demo_directml.py --demo --load ./models/revamped_XXXXXXXX_XXXXXX/final_model --viz-speed 0.05
echo.

pause 