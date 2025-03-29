"""
Training functionality for FANUC Robot ML Platform.
"""

import os
import time
import torch
import numpy as np
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from src.core.models import CustomFeatureExtractor, CustomPPO
from src.core.training.callbacks import (
    SaveModelCallback,
    TrainingMonitorCallback,
    JointLimitMonitorCallback
)
from src.utils.pybullet_utils import get_visualization_settings_from_env

def print_train_usage():
    """Print usage instructions for the train command."""
    print("Training Mode Usage:")
    print("  python fanuc_platform.py train [model_path] [steps] [options]")
    print("")
    print("Arguments:")
    print("  model_path    - Path to save the model (optional)")
    print("  steps         - Number of training steps (default: 500000)")
    print("")
    print("Options:")
    print("  --no-gui       - Disable visualization")
    print("  --eval-after   - Run evaluation after training")
    print("  --verbose      - Show detailed output")
    print("")

def create_revamped_envs(num_envs=1, viz_speed=0.0, parallel_viz=False, training_mode=True, use_env_vars=True):
    """
    Create environment(s) for training or evaluation.
    
    Args:
        num_envs: Number of parallel environments to create
        viz_speed: Visualization speed (seconds per step)
        parallel_viz: Whether to use parallel visualization
        training_mode: Whether this is used for training (affects exploration)
        use_env_vars: Whether to use environment variables for settings
        
    Returns:
        List of environment instances
    """
    # Dynamic import to avoid circular imports
    try:
        from src.core.env import JointLimitEnforcingEnv
    except ImportError as e:
        raise ImportError(f"Failed to import JointLimitEnforcingEnv: {e}")
    
    # Check for visualization settings from environment variables if requested
    env_visualize = None
    env_verbose = None
    if use_env_vars:
        try:
            from src.utils.pybullet_utils import get_visualization_settings_from_env
            env_visualize, env_verbose = get_visualization_settings_from_env()
        except Exception as e:
            print(f"Warning: Could not get visualization settings from environment: {e}")
            print("Using default values: visualize=True, verbose=False")
            env_visualize = True
            env_verbose = False
    
    # Handle import of RobotPositioningRevampedEnv with try-except
    try:
        from src.envs.robot_sim import RobotPositioningRevampedEnv
    except ImportError as e:
        print(f"Warning: Could not import RobotPositioningRevampedEnv: {e}")
        print("Falling back to RobotPositioningEnv")
        try:
            from src.envs.robot_sim import RobotPositioningEnv as RobotPositioningRevampedEnv
        except ImportError as e2:
            raise ImportError(f"Could not import robot environment classes from src.envs.robot_sim: {e2}")
    
    envs = []
    
    for i in range(num_envs):
        # Determine if visualization should be used
        use_gui = (viz_speed > 0) and (i == 0 or parallel_viz)
        # Override with environment variable if available
        if use_env_vars and env_visualize is not None and i == 0:
            use_gui = env_visualize
        
        # Determine verbosity
        use_verbose = False  # Default to false
        # Override with environment variable if available
        if use_env_vars and env_verbose is not None and i == 0:
            use_verbose = env_verbose
            
        # Create the environment with error handling
        try:
            env = RobotPositioningRevampedEnv(
                gui=use_gui,
                viz_speed=viz_speed,
                verbose=use_verbose,
                parallel_viz=parallel_viz,
                rank=i,
                offset_x=0.5 * i if parallel_viz else 0.0,
                training_mode=training_mode
            )
            
            # Apply joint limit enforcement wrapper
            env = JointLimitEnforcingEnv(env)
            
            # Add to list
            envs.append(env)
        except Exception as e:
            raise RuntimeError(f"Failed to create environment {i}: {e}")
        
    if not envs:
        raise RuntimeError("No environments were created")
        
    return envs

def train_model(model_path=None, steps=500000, visualize=True, eval_after=False, verbose=False):
    """
    Train a new FANUC robot model with DirectML acceleration.
    
    Args:
        model_path: Path to save the model (default: auto-generated)
        steps: Number of training steps
        visualize: Whether to use visualization
        eval_after: Whether to run evaluation after training
        verbose: Whether to show detailed progress
        
    Returns:
        0 if successful, 1 otherwise
    """
    # Generate a default model path if not provided
    if not model_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"models/fanuc-{timestamp}-directml"
    
    # Create models directory if it doesn't exist
    if "/" in model_path or "\\" in model_path:
        directory = os.path.dirname(model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    else:
        # If there's no directory in the path, ensure the models directory exists
        os.makedirs("models", exist_ok=True)
    
    # Set environment variables for child processes and components that read them
    # These are used by functions like get_visualization_settings_from_env
    os.environ['FANUC_VISUALIZE'] = '1' if visualize else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    # Create a variable for the model
    model = None
    
    try:
        # Ensure DirectML is available if specified
        if 'directml' in model_path.lower() or os.environ.get('FANUC_DIRECTML') == '1' or os.environ.get('USE_DIRECTML') == '1':
            from src.dml import is_available, setup_directml
            if not is_available():
                print("\nERROR: DirectML is not available in this environment.")
                print("This implementation requires an AMD GPU with DirectML support.")
                print("Please install DirectML with: pip install torch-directml")
                print("Then verify your installation with: python fanuc_platform.py install")
                return 1
                
            # Initialize DirectML
            try:
                dml_device = setup_directml()
                if verbose:
                    print(f"DirectML initialized successfully on device: {dml_device}")
            except Exception as e:
                print(f"\nERROR: DirectML initialization failed: {e}")
                print("This implementation requires DirectML to be properly initialized.")
                print("Please verify your installation with: python fanuc_platform.py install")
                if verbose:
                    import traceback
                    print(traceback.format_exc())
                return 1
            
        # Set random seed for reproducibility
        set_random_seed(42)
        
        # Create environments
        num_envs = 4  # Number of parallel environments
        viz_speed = 0.02 if visualize else 0.0
        envs = create_revamped_envs(
            num_envs=num_envs,
            viz_speed=viz_speed,
            parallel_viz=visualize,
            training_mode=True
        )
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda env=env: env for env in envs])
        
        # Normalize observations and rewards
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )
        
        # Set up policy kwargs
        policy_kwargs = {
            "features_extractor_class": CustomFeatureExtractor,
            "activation_fn": torch.nn.ReLU,
            "net_arch": [dict(pi=[256, 128, 64], vf=[256, 128, 64])],
        }
        
        # Compute appropriate batch size based on n_steps and parallel environments
        # This ensures we use full trajectories
        n_steps = 2048
        n_steps_per_env = n_steps // num_envs
        batch_size = min(64, n_steps_per_env * num_envs)
        
        # Ensure batch size is compatible with n_steps
        if n_steps_per_env * num_envs % batch_size != 0:
            # Adjust to nearest compatible batch size
            old_batch_size = batch_size
            batch_size = n_steps_per_env * num_envs // (n_steps_per_env * num_envs // batch_size)
            if verbose:
                print(f"Adjusted batch size from {old_batch_size} to {batch_size} for compatibility")
        
        # Print training parameters
        if verbose:
            print("\nTraining Parameters:")
            # Check for DirectML first, then CUDA, then fall back to CPU
            device_type = "cpu"
            if os.environ.get('USE_DIRECTML') == '1':
                try:
                    import torch_directml  # type: ignore
                    device_type = "directml"
                except ImportError:
                    print("Warning: USE_DIRECTML is set but torch_directml is not available")
                    if torch.cuda.is_available():
                        device_type = "cuda"
            elif torch.cuda.is_available():
                device_type = "cuda"
            
            print(f"Device: {device_type}")
            print(f"Learning Rate: {3e-4}")
            print(f"Timesteps: {steps}")
            print(f"n_steps: {n_steps}")
            print(f"Batch Size: {batch_size}")
            print(f"n_epochs: {10}")
            print(f"Parallel Environments: {num_envs}")
            print(f"Gamma: {0.99}")
            print(f"GAE Lambda: {0.95}")
            print(f"Clip Range: {0.2}")
            print(f"VF Coefficient: {0.5}")
            print("Architecture:", policy_kwargs["net_arch"])
            print()
        
        # Create the model timestamp for saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use original model path if provided
        if model_path:
            model_dir = model_path
        else:
            # Otherwise create a timestamped directory
            model_dir = f"./models/ppo_{timestamp}"
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Create directories for plots
        plot_dir = f"./plots/{os.path.basename(model_dir)}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Try to load existing model if requested
        loaded_model = False
        if os.path.exists(model_path):
            try:
                print(f"Loading existing model from {model_path}")
                model = CustomPPO.load(model_path, env=vec_env)
                loaded_model = True
                
                # Try to load normalization stats if available
                norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize_stats")
                if os.path.exists(norm_path):
                    print(f"Loading normalization stats from {norm_path}")
                    vec_env = VecNormalize.load(norm_path, vec_env)
                    # Don't update stats during training, we want to preserve the loaded stats
                    vec_env.training = True  # Set to true as we are continuing training
                    print("Normalization stats loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                loaded_model = False
                print("Starting with a fresh model...")
        
        # Create fresh model if not loaded
        if not loaded_model:
            print("Creating new PPO model")
            model = CustomPPO(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                normalize_advantage=True,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=verbose,
                policy_kwargs=policy_kwargs,
                device='auto',
                tensorboard_log=f"{model_dir}/tensorboard"
            )
        
        # Set up callbacks
        save_callback = SaveModelCallback(
            save_freq=steps // 10,  # Save 10 checkpoints during training
            save_path=model_dir,
            verbose=verbose
        )
        
        monitor_callback = TrainingMonitorCallback(
            log_interval=1000 if verbose else 10000,
            plot_interval=steps // 5,  # Create 5 plots during training
            plot_dir=plot_dir,
            model_dir=model_dir,
            verbose=verbose
        )
        
        joint_limit_monitor = JointLimitMonitorCallback(
            log_interval=5000,
            verbose=verbose
        )
        
        # Combine callbacks
        callbacks = [save_callback, monitor_callback, joint_limit_monitor]
        
        # Enable deterministic garbage collection
        import gc
        gc.enable()
        
        # Train the model
        print("\nStarting training...\n")
        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            log_interval=1 if verbose else 10,
        )
        
        # Save the final model
        final_model_path = os.path.join(model_dir, "final_model")
        model.save(final_model_path)
        
        # Save normalization statistics
        norm_path = os.path.join(model_dir, "vec_normalize_stats")
        vec_env.save(norm_path)
        
        print(f"\nTraining completed. Final model saved to {final_model_path}")
        
        # Run a final evaluation if requested
        if eval_after:
            print("\nRunning final evaluation...")
            from src.core.evaluation.evaluate import evaluate_model_wrapper
            evaluate_model_wrapper(
                model_path=final_model_path,
                num_episodes=5,
                visualize=visualize,
                verbose=verbose,
                max_steps=1000,
                viz_speed=viz_speed if visualize else 0.0
            )
        
        return 0
    
    except Exception as e:
        import traceback
        print(f"\nERROR: Training failed: {e}")
        if verbose:
            print(traceback.format_exc())
        return 1

def train_revamped_robot(args):
    """
    Train a robot with a consolidated, hardware-agnostic approach.
    Standardized hyperparameters and memory-efficient implementations.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    return train_model(
        model_path=args.model_path,
        steps=args.steps,
        visualize=not args.no_gui,
        eval_after=args.eval_after,
        verbose=args.verbose
    ) 