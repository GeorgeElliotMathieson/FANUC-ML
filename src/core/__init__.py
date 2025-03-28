"""
Core functionality for the FANUC robot control framework.

Contains the main implementation of the robot environment, training algorithms, 
and evaluation logic for positioning tasks.
"""

# Export key components from the positioning revamped module
try:
    from .train_robot_rl_positioning_revamped import (
        RobotPositioningRevampedEnv,
        CustomFeatureExtractor,
        JointLimitedBox,
        JointLimitEnforcingEnv,
        create_revamped_envs,
        train_revamped_robot,
        evaluate_model,
        run_evaluation_sequence,
        CustomPPO
    )
except ImportError as e:
    import sys
    print(f"Warning: Failed to import core components: {e}", file=sys.stderr)
