"""
Core functionality for the FANUC robot control framework.

Contains the main implementation of the robot environment, training algorithms, 
and evaluation logic for positioning tasks.
"""

# Export key components from the utils module
from .utils import (
    print_banner,
    print_usage,
    ensure_model_file_exists,
    is_directml_available,
    get_directml_device
)

# Import refactored modules
try:
    from . import utils
    from . import install
    from . import evaluation
    from . import testing
    from . import training
    from . import cli
    from . import env
    from . import models
except ImportError as e:
    import sys
    print(f"Warning: Failed to import refactored modules: {e}", file=sys.stderr)

# Import components from their new locations for backward compatibility
import warnings
warnings.warn(
    "The legacy import structure is deprecated. "
    "Please import directly from the refactored modules instead.",
    DeprecationWarning, 
    stacklevel=2
)

try:
    # Import from appropriate modules rather than robot_trainer
    from src.envs.robot_sim import RobotPositioningRevampedEnv as RobotPositioningEnv
    from .models.features import CustomFeatureExtractor
    from .env.action_spaces import JointLimitedBox, JointLimitEnforcingEnv
    from .training.train import create_revamped_envs as create_envs
    from .training.train import train_revamped_robot as train_robot
    from .evaluation.evaluate import evaluate_model, run_evaluation_sequence
    from .models.ppo import CustomPPO
except ImportError as e:
    import sys
    print(f"Warning: Failed to import core components: {e}", file=sys.stderr)
