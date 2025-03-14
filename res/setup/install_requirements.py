# install_requirements.py
# Run this to install all required packages

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Core libraries
install("numpy")
install("matplotlib")
install("pybullet")
install("gym")
install("pytorch")
install("stable-baselines3[extra]")
install("tensorboard")