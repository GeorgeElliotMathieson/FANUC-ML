"""
Common utilities for the FANUC Robot ML Platform.
"""

import os

def print_banner(title):
    """Print a formatted banner with a title."""
    print("\n" + "="*80)
    print(f"FANUC Robot with DirectML - {title}")
    print("="*80 + "\n")

def print_usage():
    """Print usage instructions for the script."""
    print("\nFANUC Robot ML Platform - DirectML CLI")
    print("\nAvailable commands:")
    print("  python fanuc_platform.py train    - Train a model with DirectML")
    print("  python fanuc_platform.py eval     - Evaluate a model thoroughly with DirectML")
    print("  python fanuc_platform.py test     - Run a quick test of a model with DirectML")
    print("  python fanuc_platform.py install  - Test DirectML installation")
    print("\nFor help on a specific command:")
    print("  python fanuc_platform.py train --help")
    print("  python fanuc_platform.py eval --help")
    print("  python fanuc_platform.py test --help")
    print("  python fanuc_platform.py install --help")

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
    
    # Try looking in the models directory
    model_dir_path = os.path.join("models", model_path)
    if os.path.exists(model_dir_path):
        return model_dir_path
    if os.path.exists(model_dir_path + ".pt"):
        return model_dir_path + ".pt"
    
    # Try with timestamp directories
    model_files = []
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith(".pt") or file.endswith(".zip"):
                if model_path in os.path.join(root, file):
                    model_files.append(os.path.join(root, file))
    
    if model_files:
        # Return the most recent match
        print(f"Found {len(model_files)} possible model files matching {model_path}")
        print(f"Using: {model_files[0]}")
        return model_files[0]
    
    # Neither exists
    print(f"Warning: Model file not found at {model_path}")
    return model_path

def is_directml_available():
    """Check if DirectML is available for GPU acceleration."""
    try:
        from src.dml import is_available
        return is_available()
    except ImportError:
        return False

def get_directml_device():
    """
    Get a DirectML device if available.
    
    Returns:
        DirectML device if available, None otherwise
    """
    try:
        from src.dml import get_device
        return get_device()
    except (ImportError, RuntimeError) as e:
        print(f"Error getting DirectML device: {e}")
        return None

def get_directml_settings_from_env():
    """
    Get DirectML settings from environment variables.
    
    Checks for FANUC_DIRECTML, USE_DIRECTML, and USE_GPU environment variables
    and returns whether DirectML should be used.
    
    Returns:
        bool: Whether to use DirectML
    """
    try:
        # Check for any of the environment variables that would indicate DirectML usage
        use_directml = (
            os.environ.get('FANUC_DIRECTML', '0').lower() in ('1', 'true', 'yes', 'y', 'on', 't') or
            os.environ.get('USE_DIRECTML', '0').lower() in ('1', 'true', 'yes', 'y', 'on', 't') or
            os.environ.get('USE_GPU', '0').lower() in ('1', 'true', 'yes', 'y', 'on', 't')
        )
        
        return use_directml
    except KeyError as e:
        print(f"Warning: Environment variable error: {e}")
        print("Using default value: directml=False")
        return False
    except ValueError as e:
        print(f"Warning: Could not parse environment variable value: {e}")
        print("Using default value: directml=False")
        return False
    except Exception as e:
        print(f"Warning: Error getting DirectML settings from environment: {e}")
        print("Using default value: directml=False")
        return False 