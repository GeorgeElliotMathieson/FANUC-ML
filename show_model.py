#!/usr/bin/env python3
# show_model.py
# Simple script to demonstrate a DirectML-trained model without argument parsing conflicts

import os
import sys

def main():
    """Simple wrapper to demonstrate a DirectML model"""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python show_model.py <model_path> [viz_speed] [no_gui]")
        print("Examples:")
        print("  python show_model.py ./models/ppo_directml_20250326_202801/final_model.pt")
        print("  python show_model.py ./models/ppo_directml_20250326_202801/final_model.pt 0.05")
        print("  python show_model.py ./models/ppo_directml_20250326_202801/final_model.pt 0.02 no_gui")
        sys.exit(1)
    
    # Get arguments
    model_path = sys.argv[1]
    viz_speed = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    no_gui = len(sys.argv) > 3 and sys.argv[3].lower() in ['nogui', 'no_gui', 'no-gui']
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot - DirectML Model Demonstration")
    print("Using run_directml_demo function from train_robot.py")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Visualization speed: {viz_speed}")
    print(f"GUI: {'Disabled' if no_gui else 'Enabled'}")
    print("="*80 + "\n")
    
    # Add the current directory to Python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Import and run the function
    try:
        from train_robot import run_directml_demo
        result = run_directml_demo(model_path, viz_speed, no_gui)
        return result
    except ImportError:
        print("Error: Could not import run_directml_demo from train_robot.py")
        print("Make sure train_robot.py is in the current directory.")
        return 1
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 