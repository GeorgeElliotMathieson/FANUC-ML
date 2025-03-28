#!/usr/bin/env python3
# run_directml_model.py
# Simple wrapper to run the DirectML demo from train_robot.py

import os
import sys
import argparse

def main():
    """
    Simple wrapper to run the DirectML demo from train_robot.py
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run a DirectML-trained robot model")
    parser.add_argument("model", type=str, help="Path to the model file")
    parser.add_argument("--viz-speed", type=float, default=0.02, help="Visualization speed")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    
    args = parser.parse_args()
    
    # Add project directory to python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Check if model exists
    model_path = args.model
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".pt"):
            model_path = model_path + ".pt"
        else:
            print(f"Error: Model file not found at {model_path}")
            return 1
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot - DirectML Model Runner")
    print("Using run_directml_demo from train_robot.py")
    print("="*80 + "\n")
    
    # Import the run_directml_demo function from train_robot.py
    try:
        from train_robot import run_directml_demo
        
        # Call the function
        result = run_directml_demo(
            model_path=model_path,
            viz_speed=args.viz_speed,
            no_gui=args.no_gui
        )
        
        return result
        
    except ImportError as e:
        print(f"Error importing train_robot: {e}")
        print("Make sure train_robot.py is in the current directory.")
        return 1
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 