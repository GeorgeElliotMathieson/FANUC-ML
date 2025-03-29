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
        import torch_directml
        return True
    except ImportError:
        return False

def get_directml_device():
    """
    Get a DirectML device if available.
    
    Returns:
        DirectML device if available, None otherwise
    """
    if is_directml_available():
        try:
            import torch_directml
            return torch_directml.device()
        except Exception as e:
            print(f"Error creating DirectML device: {e}")
            return None
    return None 