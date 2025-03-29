"""
FANUC Robot Machine Learning Framework

A comprehensive ML framework for training and evaluating reinforcement learning models
for FANUC robot arm positioning tasks, with special optimizations for DirectML.
"""

__version__ = "0.1.0"
__author__ = "FANUC ML Team"

# Import key DirectML functionality
from .dml import is_available as is_directml_available
from .dml import setup_directml, get_device

# Core modules are imported directly from their packages
# See src/core/__init__.py for core exports
