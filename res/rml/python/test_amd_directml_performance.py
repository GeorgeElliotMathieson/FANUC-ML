#!/usr/bin/env python3
# test_amd_directml_performance.py
# Script to test and benchmark AMD GPU performance with DirectML compared to CPU

import os
import time
import platform
import argparse
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test AMD GPU performance with DirectML for PyTorch')
parser.add_argument('--size', type=int, default=2000, help='Matrix size for benchmarking')
parser.add_argument('--iterations', type=int, default=3, help='Number of iterations for benchmarking')
args = parser.parse_args()

print("=" * 60)
print("AMD GPU Performance Test with DirectML")
print("=" * 60)
print(f"System: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"Matrix size: {args.size}x{args.size}")
print(f"Iterations: {args.iterations}")
print("=" * 60)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Try to import torch_directml
    try:
        import torch_directml
        directml_available = True
        try:
            print(f"DirectML version: {torch_directml.__version__}")
        except AttributeError:
            print("DirectML version: Not available")
        
        # Get available devices
        dml_device = torch_directml.device()
        print(f"DirectML device available: {dml_device}")
        
    except ImportError:
        directml_available = False
        print("DirectML not available. Install with: pip install torch-directml")
    
    print("=" * 60)
    
    # Define benchmark function to test performance
    def benchmark_matmul(device_type, size, iterations):
        total_time = 0
        
        if device_type == 'directml':
            device = torch_directml.device()
        else:
            device = 'cpu'
        
        # Warmup
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        
        # We don't use synchronize as it's not directly available in torch_directml
        # Using device.synchronize() would be the method but it's not available
        # Using sleep instead as a simple workaround
        if device_type == 'directml':
            # Force sync by moving data back to CPU
            _ = c.cpu()
        
        # Benchmark
        for i in range(iterations):
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Sync before timing
            if device_type == 'directml':
                # Force sync by moving data back to CPU
                _ = a.cpu()[0,0].item()
                _ = b.cpu()[0,0].item()
            
            start_time = time.time()
            c = torch.matmul(a, b)
            
            # Sync after operation to ensure timing is accurate
            if device_type == 'directml':
                # Force sync by moving data back to CPU
                _ = c.cpu()[0,0].item()
                
            end_time = time.time()
            
            iter_time = end_time - start_time
            total_time += iter_time
            print(f"Iteration {i+1}: {iter_time:.4f} seconds")
        
        avg_time = total_time / iterations
        print(f"Average time: {avg_time:.4f} seconds")
        return avg_time
    
    # Run CPU benchmark
    print("CPU Benchmark:")
    cpu_time = benchmark_matmul('cpu', args.size, args.iterations)
    
    # Run DirectML benchmark if available
    dml_time = None
    if directml_available:
        print("\nDirectML (AMD GPU) Benchmark:")
        dml_time = benchmark_matmul('directml', args.size, args.iterations)
    
    # Print comparison results
    if dml_time is not None:
        print("\nResults:")
        print("-" * 60)
        print(f"CPU time: {cpu_time:.4f} seconds")
        print(f"DirectML (GPU) time: {dml_time:.4f} seconds")
        print(f"Speedup: {cpu_time / dml_time:.2f}x")
        
        if cpu_time / dml_time < 1.2:
            print("\nWARNING: GPU is not significantly faster than CPU.")
            print("This may indicate that the GPU is not being used properly.")
            print("For AMD GPUs, make sure DirectML is configured correctly.")
        elif cpu_time / dml_time > 5:
            print("\nSuccess! Your AMD GPU is providing substantial acceleration with DirectML.")
    else:
        print("\nNo DirectML benchmark performed. Install DirectML with: pip install torch-directml")
    
except ImportError:
    print("Error: PyTorch is not installed.")
    print("Please install PyTorch and DirectML:")
    print("pip install torch")
    print("pip install torch-directml")

print("\n" + "=" * 60)
print("For more information on setting up your AMD GPU, see AMD_GPU_WINDOWS_ISSUE.md")
print("=" * 60) 