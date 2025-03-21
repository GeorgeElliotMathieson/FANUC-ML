#!/usr/bin/env python3
# test_amd_gpu_performance.py
# Script to test and benchmark AMD GPU performance compared to CPU

import os
import time
import platform
import argparse
import numpy as np

# Set useful environment variables for AMD GPUs
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test AMD GPU performance for PyTorch')
parser.add_argument('--size', type=int, default=5000, help='Matrix size for benchmarking')
parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for benchmarking')
args = parser.parse_args()

print("=" * 60)
print("AMD GPU Performance Test")
print("=" * 60)
print(f"System: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"Matrix size: {args.size}x{args.size}")
print(f"Iterations: {args.iterations}")
print("=" * 60)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for CPU-only PyTorch
    if "+cpu" in torch.__version__:
        print("WARNING: You have CPU-only PyTorch installed.")
        print("To use your AMD GPU, install PyTorch with ROCm support.")
        print("Run the install_amd_pytorch.ps1 or install_amd_pytorch.bat script.")
        print("=" * 60)
    
    # Check for CUDA and ROCm/HIP support
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    
    has_hip = hasattr(torch, 'hip')
    if has_hip and hasattr(torch.hip, 'is_available'):
        has_hip = torch.hip.is_available()
    print(f"ROCm/HIP available: {has_hip}")

    # Report ROCm version if available
    if has_hip:
        print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'}")
        print(f"Number of AMD GPUs: {torch.hip.device_count()}")
        print(f"Current device: {torch.hip.current_device()}")
        print(f"Device name: {torch.hip.get_device_name(0)}")
    
    print("=" * 60)
    
    # Define benchmark function to test performance
    def benchmark_matmul(device, size, iterations):
        total_time = 0
        
        # Warmup
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.matmul(a, b)
        
        if device != 'cpu':
            torch.cuda.synchronize() if has_cuda else torch.hip.synchronize()
        
        # Benchmark
        for i in range(iterations):
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            start_time = time.time()
            c = torch.matmul(a, b)
            if device != 'cpu':
                torch.cuda.synchronize() if has_cuda else torch.hip.synchronize()
            end_time = time.time()
            
            iter_time = end_time - start_time
            total_time += iter_time
            print(f"Iteration {i+1}: {iter_time:.4f} seconds")
        
        avg_time = total_time / iterations
        print(f"Average time: {avg_time:.4f} seconds")
        return avg_time
    
    # Run benchmarks on available devices
    print("CPU Benchmark:")
    cpu_time = benchmark_matmul('cpu', args.size, args.iterations)
    
    gpu_time = None
    
    # Test GPU if available (CUDA or HIP)
    if has_cuda:
        print("\nCUDA GPU Benchmark:")
        gpu_time = benchmark_matmul('cuda', args.size, args.iterations)
    elif has_hip:
        print("\nAMD GPU Benchmark:")
        gpu_time = benchmark_matmul('hip', args.size, args.iterations)
    
    # Print comparison results
    if gpu_time is not None:
        print("\nResults:")
        print("-" * 60)
        print(f"CPU time: {cpu_time:.4f} seconds")
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
        if cpu_time / gpu_time < 1.2:
            print("\nWARNING: GPU is not significantly faster than CPU.")
            print("This may indicate that the GPU is not being used properly.")
            print("If you have an AMD GPU, make sure PyTorch is installed with ROCm support.")
        elif cpu_time / gpu_time > 5:
            print("\nSuccess! Your GPU is providing substantial acceleration.")
    else:
        print("\nNo GPU benchmark performed. No compatible GPU found.")
    
except ImportError:
    print("Error: PyTorch is not installed.")
    print("Please install PyTorch with ROCm support for AMD GPUs:")
    print("Run the install_amd_pytorch.ps1 or install_amd_pytorch.bat script.")

print("\n" + "=" * 60)
print("For more information on setting up your AMD GPU, see AMD_GPU_USAGE.md")
print("=" * 60) 