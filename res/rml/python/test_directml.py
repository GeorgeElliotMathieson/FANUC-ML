import torch
import os
import sys
import time

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import torch_directml
    print(f"DirectML version: {torch_directml.__version__}")
    
    # Create a DirectML device
    device = torch_directml.device()
    
    # Test with a simple matrix multiplication
    size = 2000
    print(f"Testing with {size}x{size} matrix multiplication...")
    
    # CPU timing
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # DirectML timing
    a_dml = a_cpu.to(device)
    b_dml = b_cpu.to(device)
    
    # Warmup
    _ = torch.mm(a_dml, b_dml)
    
    start_time = time.time()
    c_dml = torch.mm(a_dml, b_dml)
    _ = c_dml.to('cpu')  # Synchronize
    dml_time = time.time() - start_time
    print(f"DirectML time: {dml_time:.4f} seconds")
    
    speedup = cpu_time / dml_time
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print("SUCCESS: DirectML is working and providing acceleration")
    else:
        print("WARNING: DirectML is working but not providing acceleration")
        
except ImportError:
    print("ERROR: torch_directml not installed")
    print("Please run 'pip install torch-directml' to install it")
