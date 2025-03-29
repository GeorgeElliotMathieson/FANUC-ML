"""
Core functionality for the FANUC robot control framework.

Contains the main implementation of the robot environment, training algorithms, 
and evaluation logic for positioning tasks.
"""

# Export key components from the robot trainer module
try:
    from .robot_trainer import (
        RobotPositioningEnv,
        CustomFeatureExtractor,
        JointLimitedBox,
        JointLimitEnforcingEnv,
        create_envs,
        train_robot,
        evaluate_model,
        run_evaluation_sequence,
        CustomPPO
    )
except ImportError as e:
    import sys
    print(f"Warning: Failed to import core components: {e}", file=sys.stderr)
