#!/usr/bin/env python3
# analyze_model.py
# Utility to analyze and diagnose model files

import os
import sys
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PyTorch model files")
    parser.add_argument("model_path", help="Path to the model file to analyze")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    
    print(f"\nAnalyzing model file: {model_path}")
    print("="*80)
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"ERROR: File not found: {model_path}")
        return 1
    
    # Print file size
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size/1024:.2f} KB ({file_size} bytes)")
    
    # Try loading with different methods
    try:
        print("\nAttempting to load with torch.load to CPU...")
        data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check what type of data we have
        print(f"Data type: {type(data)}")
        
        # If it's a dictionary, print the keys
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            
            # Check for common keys
            for key in ['model', 'policy_state_dict', 'optimizer_state_dict', 'model_state_dict']:
                if key in data:
                    print(f"Found '{key}' in the dictionary.")
                    if isinstance(data[key], dict):
                        print(f"  '{key}' is a dictionary with keys: {list(data[key].keys())}")
        
        # If it's a nn.Module, print structure
        if hasattr(data, 'state_dict'):
            print("\nModel has state_dict method - it's likely a nn.Module")
            state_dict = data.state_dict()
            print(f"State dict keys: {list(state_dict.keys())}")
            
            # Print some parameter shapes
            print("\nSample parameter shapes:")
            for i, (key, param) in enumerate(state_dict.items()):
                print(f"  {key}: {param.shape}")
                if i >= 9:  # Show only first 10
                    print(f"  ... and {len(state_dict)-10} more parameters.")
                    break
    
    except Exception as e:
        print(f"Error loading with torch.load: {e}")
    
    try:
        print("\nAttempting to check file format...")
        with open(model_path, 'rb') as f:
            magic_number = f.read(8)
            # Check for PyTorch format
            if magic_number.startswith(b'PK\x03\x04'):
                print("File appears to be a ZIP archive (TorchScript model)")
            # Try other formats
            elif magic_number.startswith(b'\x80\x02'):
                print("File appears to be a Pickle file (standard PyTorch save)")
            # Try ONNX format
            elif magic_number.startswith(b'ONNX'):
                print("File appears to be an ONNX model")
            else:
                print(f"Unknown file format. Magic number: {magic_number.hex()}")
    
    except Exception as e:
        print(f"Error checking file format: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 