"""
Training functionality for FANUC Robot ML Platform.
"""

import os
import time
import torch
import torch.nn as nn  # Import nn module
import numpy as np
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import warnings

from src.core.models import CustomFeatureExtractor, CustomPPO
from src.core.training.callbacks import (
    SaveModelCallback,
    TrainingMonitorCallback,
    JointLimitMonitorCallback
)
from src.utils.pybullet_utils import get_visualization_settings_from_env, get_directml_settings_from_env

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

def create_revamped_envs(n_envs=1, viz_speed=0.0, parallel_viz=False, training_mode=True, use_env_vars=True, **kwargs):
    """
    Create environment(s) for training or evaluation.
    
    Args:
        n_envs: Number of parallel environments to create
        viz_speed: Visualization speed (seconds per step)
        parallel_viz: Whether to use parallel visualization
        training_mode: Whether this is used for training (affects exploration)
        use_env_vars: Whether to use environment variables for settings
        **kwargs: Additional arguments to pass to the environment
        
    Returns:
        Vector environment with multiple environments
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
    
    env_list = []
    
    # Calculate grid dimensions for robot placement
    # Determine the grid size based on the number of environments
    import math
    grid_size = math.ceil(math.sqrt(n_envs))  # Create a square grid
    
    # Define offset distance between robots (increased for more spacing)
    offset_distance = 2.5  # 2.5 meters between robots
    
    for i in range(n_envs):
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
            
        # Calculate grid position (row, column) from index
        if parallel_viz:
            row = i // grid_size
            col = i % grid_size
            offset_x = col * offset_distance
            offset_y = row * offset_distance
        else:
            offset_x = 0.0
            offset_y = 0.0
            
        # Create the environment with error handling
        try:
            # Check the accepted parameters for RobotPositioningRevampedEnv
            # and only pass those that are actually accepted
            env_params = {
                'gui': use_gui,
                'viz_speed': viz_speed,
                'verbose': use_verbose,
                'training_mode': training_mode
            }
            
            # Add any additional arguments passed to this function
            env_params.update(kwargs)
            
            # Create the environment with appropriate parameters
            env = RobotPositioningRevampedEnv(**env_params)
            
            # Apply joint limit enforcement wrapper
            env = JointLimitEnforcingEnv(env)
            
            # Add to list
            env_list.append(env)
        except Exception as e:
            raise RuntimeError(f"Failed to create environment {i}: {e}")
        
    if not env_list:
        raise RuntimeError("No environments were created")
    
    # Adjust camera view to accommodate all robots if parallel visualization is enabled
    if parallel_viz and viz_speed > 0:
        from src.utils.pybullet_utils import adjust_camera_for_robots
        try:
            # Get the client from the first environment
            client_id = env_list[0].client_id
            # Adjust camera based on number of environments and grid layout
            adjust_camera_for_robots(client_id, num_robots=n_envs, workspace_size=0.8, grid_layout=True, grid_size=grid_size)
        except Exception as e:
            print(f"Warning: Could not adjust camera for multiple robots: {e}")
    
    # Create a vector environment
    vec_env = DummyVecEnv([lambda env=env: env for env in env_list])
    
    return vec_env

def train_model(
    model_dir, 
    num_timesteps, 
    seed=42,
    use_directml=None,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=1024,  # Increased from 64 to 1024 for better batch processing
    n_epochs=10,
    n_envs=1,  # Changed default from 8 to 1
    env_kwargs=None,
    policy_kwargs=None,
    save_freq=1000,
    eval_freq=0,
    eval_episodes=10,
    model_class="PPO",
    verbose=False,
    viz_speed=0.0,
    use_curriculum=False,
    log_tensorboard=True,
):
    """
    Train a reinforcement learning model.
    
    Args:
        model_dir: Directory to save the model
        num_timesteps: Number of timesteps to train for
        seed: Random seed
        use_directml: Whether to use DirectML (AMD GPU acceleration)
        learning_rate: Learning rate
        n_steps: Number of steps per rollout
        batch_size: Batch size
        n_epochs: Number of epochs
        n_envs: Number of parallel environments
        env_kwargs: Keyword arguments for environment creation
        policy_kwargs: Keyword arguments for policy creation
        save_freq: Frequency of saving model checkpoints
        eval_freq: Frequency of evaluating the model
        eval_episodes: Number of episodes to evaluate the model
        model_class: Name of the model class to use
        verbose: Whether to print verbose output
        viz_speed: Visualization speed (0.0 for no visualization)
        use_curriculum: Whether to use curriculum learning
        log_tensorboard: Whether to log metrics to TensorBoard
        
    Returns:
        Trained model
    """
    # Set up environment variables
    os.environ['TRAIN_VERBOSE'] = '1' if verbose else '0'
    os.environ['TRAINING_MODE'] = '1'  # Always set training mode to true during training
    
    # Configure DirectML if requested
    device = None
    if use_directml is None:
        # Check if DirectML is requested via environment variables
        use_directml, dml_settings = get_directml_settings_from_env()
    
    # Override from function parameter if explicitly set
    if use_directml:
        if verbose:
            print("Setting up DirectML for GPU acceleration...")
        
        try:
            # Import DirectML setup function
            from src.dml import setup_directml
            
            # Initialize DirectML
            device = setup_directml()
            
            if device is not None:
                print("DirectML setup complete, using device:", device)
                
                # Set number of threads for PyTorch with DirectML
                import torch
                
                # Optimize CPU thread usage with DirectML
                torch.set_num_threads(4)  # Limit CPU threads during GPU training
                
                # Auto-adjust batch size for better GPU utilization
                if batch_size < 128:
                    original_batch_size = batch_size
                    batch_size = max(batch_size, 512)  # Ensure minimum batch size of 512 for GPU
                    if verbose and original_batch_size != batch_size:
                        print(f"Increasing batch size from {original_batch_size} to {batch_size} for better GPU utilization")
                
                # Auto-adjust n_steps for better GPU batching
                if n_steps < 1024:
                    original_n_steps = n_steps
                    n_steps = max(n_steps, 2048)  # Ensure minimum steps of 2048 for GPU
                    if verbose and original_n_steps != n_steps:
                        print(f"Increasing n_steps from {original_n_steps} to {n_steps} for better GPU batching")
                
                # Configure GPU-optimized learning
                if "DIRECTML_ENABLE_TENSOR_CORES" not in os.environ:
                    os.environ["DIRECTML_ENABLE_TENSOR_CORES"] = "1"
                if "DIRECTML_GPU_TRANSFER_OPTIMIZATION" not in os.environ:
                    os.environ["DIRECTML_GPU_TRANSFER_OPTIMIZATION"] = "1"
                if "DIRECTML_ENABLE_OPTIMIZATION" not in os.environ:
                    os.environ["DIRECTML_ENABLE_OPTIMIZATION"] = "1"
            else:
                print("DirectML device creation failed, falling back to CPU")
                use_directml = False
        except ImportError as e:
            print(f"Failed to import DirectML module: {e}")
            print("Falling back to CPU training")
            use_directml = False
        except Exception as e:
            print(f"DirectML setup failed: {e}")
            print("Falling back to CPU training")
            use_directml = False
            
    # Create environment vector
    if verbose:
        print(f"Creating {n_envs} parallel training environments...")
    
    # Set default environment kwargs if not provided
    if env_kwargs is None:
        env_kwargs = {}
    
    # Add training_mode flag to env_kwargs if not present
    if 'training_mode' not in env_kwargs:
        env_kwargs['training_mode'] = True
        
    # Check if curriculum learning is requested
    if use_curriculum:
        if verbose:
            print("Using curriculum learning.")
        env_kwargs['use_curriculum'] = True
    
    # Prevent duplicate viz_speed parameter
    if 'viz_speed' in env_kwargs:
        # Create the training environments
        envs = create_revamped_envs(n_envs=n_envs, **env_kwargs)
    else:
        # Create the training environments
        envs = create_revamped_envs(n_envs=n_envs, viz_speed=viz_speed, **env_kwargs)
    
    print(f"Created vector environment with {n_envs} parallel environments")
    
    # Set up directories
    os.makedirs(model_dir, exist_ok=True)
    if verbose:
        print(f"Model will be saved to: {model_dir}")
    
    # Set up an eval environment if requested
    eval_env = None
    if eval_freq > 0:
        if verbose:
            print(f"Creating evaluation environment...")
        eval_env = create_revamped_envs(
            n_envs=1, viz_speed=viz_speed, parallel_viz=False, training_mode=False, **env_kwargs or {}
        )
    
    # Set up policy keyword arguments if not provided
    if policy_kwargs is None:
        policy_kwargs = {}
    
    # Set up model class
    if model_class == "PPO":
        if verbose:
            print("Using DirectML-optimized PPO" if use_directml else "Using standard PPO")
            
        # Import necessary modules
        from src.core.models.ppo import CustomPPO
        from src.core.models.features import CustomFeatureExtractor
        from stable_baselines3.common.policies import ActorCriticPolicy
        
        # Filter out the DirectML CPU fallback warning
        warnings.filterwarnings("ignore", message=".*not currently supported on the DML backend.*")
        
        # Set up GPU-specific policy settings
        if use_directml and device is not None:
            # Update policy kwargs for better GPU utilization
            if 'net_arch' not in policy_kwargs:
                # Use optimally balanced network architecture
                policy_kwargs['net_arch'] = dict(pi=[512, 384, 256], vf=[512, 384, 256])
                
            # Add feature extractor for GPU
            policy_kwargs['features_extractor_class'] = CustomFeatureExtractor
            policy_kwargs['features_extractor_kwargs'] = {'features_dim': 384}  # Optimal feature dimension
                
            # Set device explicitly
            policy_kwargs['device'] = device
                
            # Setup tensorboard for better monitoring
            if log_tensorboard:
                tensorboard_log = os.path.join(model_dir, "tb_logs")
                os.makedirs(tensorboard_log, exist_ok=True)
                if verbose:
                    print("Enabling TensorBoard logging")
            else:
                tensorboard_log = None
                
            # Create model with optimized parameters
            if verbose:
                print("Creating model...")
                if device:
                    print(f"Using {device} device")
                
                model = CustomPPO(
                    policy=ActorCriticPolicy,
                    env=envs,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=0.2,
                    normalize_advantage=True,  # Keep normalization but warnings will be filtered
                    ent_coef=0.01,  # Slightly increased for better exploration
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    verbose=1 if verbose else 0,
                    tensorboard_log=tensorboard_log,
                    policy_kwargs=policy_kwargs,
                    seed=seed,
                    device=device
                )
        else:
            # Standard CPU setup
            if log_tensorboard:
                tensorboard_log = os.path.join(model_dir, "tb_logs")
                os.makedirs(tensorboard_log, exist_ok=True)
                if verbose:
                    print("Enabling TensorBoard logging")
            else:
                tensorboard_log = None
                
            if verbose:
                print("Creating model...")
                
            model = CustomPPO(
                policy=ActorCriticPolicy,
                env=envs,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                verbose=1 if verbose else 0,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                seed=seed
            )
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    if verbose:
        print("Model created. Starting training...")
    
    # Set up the callback for saving checkpoints
    callbacks = []
    if save_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // n_envs,  # Convert to per-environment steps
            save_path=model_dir,
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=1 if verbose else 0,
        )
        callbacks.append(checkpoint_callback)
    
    # Set up callback for evaluation
    if eval_freq > 0 and eval_episodes > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(model_dir, "eval_logs"),
            eval_freq=eval_freq // n_envs,
            deterministic=True,
            render=False,
            n_eval_episodes=eval_episodes,
            verbose=1 if verbose else 0,
        )
        callbacks.append(eval_callback)
    
    # Combine callbacks
    if callbacks:
        callback = CallbackList(callbacks)
    else:
        callback = None
    
    # Train the model
    if verbose:
        print(f"Starting training for {num_timesteps} timesteps...")
        print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Steps: {n_steps}")
        if use_directml:
            print("Training with DirectML GPU acceleration")
        else:
            print("Training on CPU")
    
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=num_timesteps,
            callback=callback,
            tb_log_name="training",
            progress_bar=verbose,
        )
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Training completed in {elapsed_time:.2f} seconds")
            print(f"Average FPS: {num_timesteps / elapsed_time:.1f}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    
    if verbose:
        print(f"Final model saved to {final_model_path}")
    
    # Clean up
    if hasattr(envs, 'close'):
        envs.close()
    
    if 'eval_env' in locals() and hasattr(eval_env, 'close'):
        eval_env.close()
    
    return model

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
        model_dir=args.model_path,
        num_timesteps=args.steps,
        seed=args.seed,
        use_directml=args.use_directml,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_envs=args.n_envs,
        env_kwargs=args.env_kwargs,
        policy_kwargs=args.policy_kwargs,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        model_class=args.model_class,
        verbose=args.verbose,
        viz_speed=args.viz_speed,
        use_curriculum=args.use_curriculum,
        log_tensorboard=args.log_tensorboard
    ) 