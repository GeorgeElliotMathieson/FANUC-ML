#!/usr/bin/env python3
# directml_utils.py
# Utility functions for working with DirectML models

import os
import sys
import torch
import numpy as np
from typing import Optional, Union, Tuple, Any

def block_argparse():
    """
    Block argument parsing in imported modules by temporarily
    replacing sys.argv with an empty list.
    """
    # Save original arguments
    original_argv = sys.argv.copy()
    # Replace with minimal arguments to prevent parsing errors
    sys.argv = [sys.argv[0]]
    return original_argv

def restore_argparse(original_argv):
    """
    Restore the original command line arguments.
    """
    sys.argv = original_argv

def setup_directml() -> Optional[torch.device]:
    """
    Set up DirectML environment and verify its functionality.
    Returns the DirectML device if successful, None otherwise.
    """
    try:
        # Configure environment variables for DirectML
        os.environ["PYTORCH_DIRECTML_VERBOSE"] = "1"
        os.environ["DIRECTML_ENABLE_OPTIMIZATION"] = "1"
        os.environ["USE_DIRECTML"] = "1"
        os.environ["USE_GPU"] = "1"
        
        # Import DirectML
        import torch_directml
        
        # Check for available DirectML devices
        device_count = torch_directml.device_count()
        if device_count == 0:
            raise RuntimeError("No DirectML devices detected")
        
        # Create a DirectML device
        dml_device = torch_directml.device()
        print(f"DirectML devices available: {device_count}")
        print(f"Using DirectML device: {dml_device}")
        
        # Create a test tensor on the DirectML device to verify it works
        test_tensor = torch.ones((2, 3), device=dml_device)
        # Access the tensor to force execution on GPU
        _ = test_tensor.cpu().numpy()
        
        print("✓ DirectML acceleration active and verified")
        print("✓ Test tensor created successfully on GPU")
        
        return dml_device
        
    except ImportError as e:
        print(f"ERROR: DirectML package not found: {e}")
        print("AMD GPU acceleration will not be available.")
        print("To install DirectML support, run: pip install torch-directml")
        return None
    except Exception as e:
        import traceback
        print(f"ERROR initializing DirectML: {e}")
        print(traceback.format_exc())
        return None

def patch_directml_ppo():
    """
    Add the predict method to the DirectMLPPO class for compatibility.
    This is a monkey patch to make the DirectML model compatible with
    the standard PPO interface.
    """
    # Block argument parsing
    original_argv = block_argparse()
    
    try:
        # Import the DirectML PPO class
        from train_robot_rl_ppo_directml import DirectMLPPO
        
        # Add predict method if it doesn't exist
        if not hasattr(DirectMLPPO, 'predict'):
            def predict(self, observation, deterministic=True):
                """
                Get an action prediction from the model.
                
                Args:
                    observation: The observation from the environment
                    deterministic: Whether to use deterministic actions
                    
                Returns:
                    action: The action to take
                    _: Placeholder for state (None)
                """
                # Convert observation to tensor
                if isinstance(observation, np.ndarray):
                    observation = torch.as_tensor(observation, device=self.device).unsqueeze(0)
                
                # Get action from the model
                with torch.no_grad():
                    action, _, _ = self.policy_network.get_action(observation, deterministic=deterministic)
                
                # Convert to numpy and return
                action_np = action.cpu().numpy().flatten()
                return action_np, None
            
            # Add method to class
            setattr(DirectMLPPO, 'predict', predict)
            print("✓ Added predict method to DirectMLPPO class")
        
        # Add to_device method if it doesn't exist
        if not hasattr(DirectMLPPO, 'to_device'):
            def to_device(self, device):
                """
                Move the model to the specified device.
                
                Args:
                    device: The device to move to
                """
                # Store the new device
                self.device = device
                
                # Move policy network to device
                if hasattr(self, 'policy_network'):
                    self.policy_network = self.policy_network.to(device)
                
                # Move value network to device if separate
                if hasattr(self, 'value_network'):
                    self.value_network = self.value_network.to(device)
                
                # Move other networks if they exist
                if hasattr(self, 'feature_extractor'):
                    self.feature_extractor = self.feature_extractor.to(device)
                
                return self
            
            # Add method to class
            setattr(DirectMLPPO, 'to_device', to_device)
            print("✓ Added to_device method to DirectMLPPO class")
    
    except ImportError as e:
        print(f"Warning: Could not import DirectMLPPO class: {e}")
    except Exception as e:
        import traceback
        print(f"Error patching DirectMLPPO: {e}")
        print(traceback.format_exc())
    
    finally:
        # Restore original arguments
        restore_argparse(original_argv)

def load_directml_model(model_path: str, env: Any, device: torch.device) -> Optional[Any]:
    """
    Load a DirectML model with proper error handling.
    
    Args:
        model_path: Path to the model file
        env: Environment object to use with the model
        device: DirectML device to use
        
    Returns:
        The loaded model or None if loading failed
    """
    # Block argument parsing in imported modules
    original_argv = block_argparse()
    
    try:
        # Apply patches
        patch_directml_ppo()
        
        # Import DirectML-specific modules
        from train_robot_rl_ppo_directml import DirectMLPPO
        
        # Create a DirectML model with CPU device first
        # This avoids issues with DirectML device in torch.load
        model = DirectMLPPO(
            env=env,
            device=torch.device("cpu"),  # Start with CPU
            verbose=True
        )
        
        print(f"Loading model from {model_path} to CPU first...")
        
        try:
            # Load the model to CPU first
            model.load(model_path)
            print("✓ Model loaded to CPU successfully")
            
            # Now manually move model to DirectML device
            if device is not None:
                print(f"Moving model to DirectML device: {device}")
                model.to_device(device)
                print("✓ Model transferred to DirectML device")
        except Exception as e:
            import traceback
            print(f"Error in model loading stage: {e}")
            print(traceback.format_exc())
            return None
            
        return model
        
    except ImportError as e:
        print(f"ERROR: Required modules not found: {e}")
        print("Make sure all dependencies are installed.")
        return None
    except Exception as e:
        import traceback
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        return None
    finally:
        # Restore original arguments
        restore_argparse(original_argv)

def is_directml_model(model_path: str) -> bool:
    """
    Check if a model was trained with DirectML based on filename pattern.
    """
    return "directml" in model_path.lower()

def ensure_model_file_exists(model_path: str) -> str:
    """
    Check if model file exists and handle .pt extension if needed.
    Returns the corrected path.
    """
    if not model_path:
        print("ERROR: No model path specified.")
        return model_path
        
    # Check if the file exists as is
    if os.path.exists(model_path):
        return model_path
        
    # Try with .pt extension
    if os.path.exists(model_path + ".pt"):
        model_path += ".pt"
        print(f"Using model file with .pt extension: {model_path}")
        return model_path
    
    # Neither exists
    print(f"Warning: Model file not found at {model_path}")
    return model_path

def initialize_pybullet(render: bool = True) -> int:
    """
    Initialize PyBullet and return the client ID.
    This ensures we set up the shared client correctly.
    
    Args:
        render: Whether to use GUI mode
        
    Returns:
        PyBullet client ID
    """
    # Block argument parsing
    original_argv = block_argparse()
    
    try:
        # Import PyBullet safely
        import pybullet as p
        from res.rml.python.train_robot_rl_positioning import get_shared_pybullet_client
        
        # Get a shared client
        client_id = get_shared_pybullet_client(render=render)
        print(f"Initialized PyBullet shared client with ID: {client_id}")
        
        return client_id
    
    except ImportError as e:
        print(f"ERROR: Could not import PyBullet or related modules: {e}")
        return -1
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to initialize PyBullet: {e}")
        print(traceback.format_exc())
        return -1
    finally:
        # Restore original arguments
        restore_argparse(original_argv)

def safe_import_robot_env(render: bool = True, viz_speed: float = 0.02):
    """
    Safely import robot environment while avoiding argparse conflicts.
    
    Args:
        render: Whether to render the environment
        viz_speed: Visualization speed
        
    Returns:
        Tuple of (env_class, pybullet_module)
    """
    # Block argument parsing in imported modules
    original_argv = block_argparse()
    
    try:
        # Initialize PyBullet first (sets up shared client)
        client_id = initialize_pybullet(render=render)
        if client_id < 0:
            return None, None
            
        # Import modules
        import pybullet
        from res.rml.python.train_robot_rl_positioning_revamped import RobotPositioningRevampedEnv
        
        return RobotPositioningRevampedEnv, pybullet
        
    except ImportError as e:
        print(f"ERROR: Required modules not found: {e}")
        return None, None
    except Exception as e:
        import traceback
        print(f"Error importing robot environment: {e}")
        print(traceback.format_exc())
        return None, None
    finally:
        # Restore original arguments
        restore_argparse(original_argv) 