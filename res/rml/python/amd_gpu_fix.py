#!/usr/bin/env python3
# amd_gpu_fix.py
# Diagnostic tool for AMD GPU compatibility with PyTorch

import sys
import os
import platform
import subprocess
import importlib.util
from typing import Dict, Any, List, Optional, Tuple

def colored(text: str, color: str) -> str:
    """Add color to console output if on Linux/Mac."""
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'end': '\033[0m'
    }
    
    # Only use colors on Linux/Mac or when using Windows Terminal
    if platform.system() != 'Windows' or os.environ.get('WT_SESSION'):
        return f"{colors.get(color, '')}{text}{colors['end']}"
    return text

def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    return importlib.util.find_spec(module_name) is not None

def get_pytorch_info() -> Dict[str, Any]:
    """Get information about the PyTorch installation."""
    result = {
        'installed': False,
        'version': None,
        'cuda_available': False,
        'rocm_available': False,
        'device_count': 0,
        'device_names': [],
        'rocm_version': None,
        'is_cpu_only': False
    }
    
    if not check_import('torch'):
        return result
    
    import torch
    result['installed'] = True
    result['version'] = torch.__version__
    
    # Check for CPU-only build
    if '+cpu' in torch.__version__:
        result['is_cpu_only'] = True
    
    # Check for CUDA support
    result['cuda_available'] = torch.cuda.is_available()
    if result['cuda_available']:
        result['device_count'] = torch.cuda.device_count()
        result['device_names'] = [torch.cuda.get_device_name(i) for i in range(result['device_count'])]
    
    # Check for ROCm/HIP support (AMD GPUs)
    has_hip_attr = hasattr(torch, 'hip')
    if has_hip_attr:
        if hasattr(torch.hip, 'is_available'):
            result['rocm_available'] = torch.hip.is_available()
            
            if result['rocm_available']:
                result['device_count'] = torch.hip.device_count()
                result['device_names'] = [torch.hip.get_device_name(i) for i in range(result['device_count'])]
                
                if hasattr(torch.version, 'hip'):
                    result['rocm_version'] = torch.version.hip
    
    return result

def check_gpu_hardware() -> Dict[str, Any]:
    """Detect GPU hardware on the system."""
    result = {
        'nvidia_gpus': [],
        'amd_gpus': [],
        'has_nvidia': False,
        'has_amd': False
    }
    
    if platform.system() == 'Windows':
        try:
            # Use PowerShell to get GPU info
            cmd = ["powershell", "-Command", 
                   "Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-List"]
            output = subprocess.check_output(cmd, text=True)
            
            current_gpu = {}
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    if current_gpu and 'name' in current_gpu:
                        name = current_gpu['name'].lower()
                        if 'nvidia' in name:
                            result['nvidia_gpus'].append(current_gpu)
                            result['has_nvidia'] = True
                        elif 'amd' in name or 'radeon' in name:
                            result['amd_gpus'].append(current_gpu)
                            result['has_amd'] = True
                        current_gpu = {}
                    continue
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == 'name':
                        current_gpu['name'] = value
                    elif key == 'adapterram':
                        # Convert to GB
                        try:
                            ram_bytes = int(value)
                            current_gpu['vram_gb'] = ram_bytes / (1024**3)
                        except ValueError:
                            current_gpu['vram_gb'] = None
                    elif key == 'driverversion':
                        current_gpu['driver'] = value
            
            # Add the last GPU if there is one
            if current_gpu and 'name' in current_gpu:
                name = current_gpu['name'].lower()
                if 'nvidia' in name:
                    result['nvidia_gpus'].append(current_gpu)
                    result['has_nvidia'] = True
                elif 'amd' in name or 'radeon' in name:
                    result['amd_gpus'].append(current_gpu)
                    result['has_amd'] = True
                    
        except (subprocess.SubprocessError, OSError):
            pass
            
    elif platform.system() == 'Linux':
        try:
            # Check for NVIDIA GPUs using nvidia-smi
            try:
                nvidia_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                                       '--format=csv,noheader'], text=True)
                for line in nvidia_output.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            name, memory, driver = parts[0], parts[1], parts[2]
                            try:
                                memory_value = float(memory.split()[0])
                                memory_unit = memory.split()[1].lower()
                                if memory_unit == 'mib':
                                    vram_gb = memory_value / 1024
                                else:
                                    vram_gb = memory_value
                            except (ValueError, IndexError):
                                vram_gb = None
                                
                            result['nvidia_gpus'].append({
                                'name': name,
                                'vram_gb': vram_gb,
                                'driver': driver
                            })
                            result['has_nvidia'] = True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
            # Check for AMD GPUs using rocm-smi or lspci
            try:
                try:
                    # Try rocm-smi first
                    rocm_output = subprocess.check_output(['rocm-smi', '--showproductname'], text=True)
                    for line in rocm_output.strip().split('\n'):
                        if 'GPU' in line and 'Product name' in line:
                            name = line.split(':', 1)[1].strip()
                            result['amd_gpus'].append({
                                'name': name,
                                'vram_gb': None,
                                'driver': None
                            })
                            result['has_amd'] = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Fallback to lspci
                    lspci_output = subprocess.check_output(['lspci', '-v'], text=True)
                    for line in lspci_output.lower().split('\n'):
                        if 'vga' in line and ('amd' in line or 'radeon' in line or 'ati' in line):
                            name = line.split(':', 1)[1].strip()
                            result['amd_gpus'].append({
                                'name': name,
                                'vram_gb': None,
                                'driver': None
                            })
                            result['has_amd'] = True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        except Exception:
            pass
            
    return result

def check_environment_variables() -> Dict[str, Any]:
    """Check environment variables related to AMD GPU usage."""
    result = {
        'hip_visible_devices': os.environ.get('HIP_VISIBLE_DEVICES'),
        'pytorch_hip_alloc_conf': os.environ.get('PYTORCH_HIP_ALLOC_CONF'),
        'rocm_visible_devices': os.environ.get('ROCM_VISIBLE_DEVICES'),
        'is_configured': False
    }
    
    # Check if the essential variables are set properly
    if result['hip_visible_devices'] == '0' and result['pytorch_hip_alloc_conf'] == 'max_split_size_mb:128':
        result['is_configured'] = True
        
    return result

def print_section_header(title: str) -> None:
    """Print a section header with formatting."""
    print("\n" + "=" * 60)
    print(colored(f" {title}", 'cyan'))
    print("=" * 60)

def print_diagnostic_results(pytorch_info: Dict[str, Any], 
                           gpu_hardware: Dict[str, Any],
                           env_vars: Dict[str, Any]) -> None:
    """Print diagnostic results in a user-friendly format."""
    print_section_header("PyTorch Installation")
    
    if pytorch_info['installed']:
        print(f"PyTorch version: {colored(pytorch_info['version'], 'green')}")
        
        if pytorch_info['is_cpu_only']:
            print(colored("⚠️ CPU-only version detected!", 'yellow'))
            print("   This version cannot use any GPU acceleration.")
        
        # CUDA information
        if pytorch_info['cuda_available']:
            print(f"CUDA: {colored('Available', 'green')}")
            print(f"CUDA devices: {pytorch_info['device_count']}")
            for i, name in enumerate(pytorch_info['device_names']):
                print(f"  {i}: {name}")
        else:
            print(f"CUDA: {colored('Not available', 'yellow')}")
        
        # ROCm information
        if pytorch_info['rocm_available']:
            print(f"ROCm/HIP: {colored('Available', 'green')}")
            if pytorch_info['rocm_version']:
                print(f"ROCm version: {pytorch_info['rocm_version']}")
            print(f"ROCm devices: {pytorch_info['device_count']}")
            for i, name in enumerate(pytorch_info['device_names']):
                print(f"  {i}: {name}")
        else:
            print(f"ROCm/HIP: {colored('Not available', 'yellow')}")
    else:
        print(colored("❌ PyTorch is not installed!", 'red'))
    
    print_section_header("GPU Hardware")
    
    if gpu_hardware['has_nvidia']:
        print(f"NVIDIA GPUs: {colored(len(gpu_hardware['nvidia_gpus']), 'green')}")
        for i, gpu in enumerate(gpu_hardware['nvidia_gpus']):
            print(f"  {i}: {gpu['name']}")
            if gpu.get('vram_gb'):
                print(f"     VRAM: {gpu['vram_gb']:.1f} GB")
            if gpu.get('driver'):
                print(f"     Driver: {gpu['driver']}")
    else:
        print("NVIDIA GPUs: None detected")
    
    if gpu_hardware['has_amd']:
        print(f"AMD GPUs: {colored(len(gpu_hardware['amd_gpus']), 'green')}")
        for i, gpu in enumerate(gpu_hardware['amd_gpus']):
            print(f"  {i}: {gpu['name']}")
            if gpu.get('vram_gb'):
                print(f"     VRAM: {gpu['vram_gb']:.1f} GB")
            if gpu.get('driver'):
                print(f"     Driver: {gpu['driver']}")
    else:
        print("AMD GPUs: None detected")
    
    print_section_header("Environment Variables")
    
    if env_vars['is_configured']:
        print(colored("✓ Environment variables are correctly configured", 'green'))
    else:
        print(colored("⚠️ Environment variables are not optimally configured", 'yellow'))
    
    print(f"HIP_VISIBLE_DEVICES: {env_vars['hip_visible_devices'] or 'Not set'}")
    print(f"PYTORCH_HIP_ALLOC_CONF: {env_vars['pytorch_hip_alloc_conf'] or 'Not set'}")
    
    print_section_header("Diagnosis")
    
    # Check for AMD GPU with CPU-only PyTorch
    if gpu_hardware['has_amd'] and pytorch_info['is_cpu_only']:
        print(colored("❌ ISSUE DETECTED: AMD GPU present but PyTorch is CPU-only", 'red'))
        print("\nYour system has an AMD GPU, but you've installed a CPU-only version of PyTorch.")
        print("This means your GPU is not being used for acceleration.")
        
        print("\n" + colored("SOLUTION:", 'green'))
        print("1. Uninstall the current PyTorch version:")
        print("   pip uninstall -y torch torchvision torchaudio")
        print("2. Install PyTorch with ROCm support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6")
        print("3. Set required environment variables:")
        print("   set HIP_VISIBLE_DEVICES=0")
        print("   set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128")
        print("\nFor convenience, we've provided scripts to do this for you:")
        print("- PowerShell: .\\install_amd_pytorch.ps1")
        print("- Batch: install_amd_pytorch.bat")
        
    # Check for AMD GPU with PyTorch that has ROCm but not configured
    elif gpu_hardware['has_amd'] and not pytorch_info['is_cpu_only'] and not pytorch_info['rocm_available']:
        print(colored("⚠️ ISSUE DETECTED: AMD GPU present but ROCm/HIP support not available", 'yellow'))
        print("\nYour system has an AMD GPU, but PyTorch doesn't have ROCm/HIP support enabled.")
        
        print("\n" + colored("SOLUTION:", 'green'))
        print("1. Install PyTorch with ROCm support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6")
        print("2. Set required environment variables:")
        print("   set HIP_VISIBLE_DEVICES=0")
        print("   set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128")
        
    # Check for AMD GPU with PyTorch ROCm but not environment variables
    elif gpu_hardware['has_amd'] and not pytorch_info['is_cpu_only'] and pytorch_info['rocm_available'] and not env_vars['is_configured']:
        print(colored("⚠️ ISSUE DETECTED: Environment variables not optimally configured", 'yellow'))
        print("\nYour PyTorch installation supports ROCm/HIP, but environment variables are not set for optimal performance.")
        
        print("\n" + colored("SOLUTION:", 'green'))
        print("Set the following environment variables:")
        print("set HIP_VISIBLE_DEVICES=0")
        print("set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128")
        print("\nFor permanent configuration, run as Administrator:")
        print("- PowerShell: .\\set_permanent_amd_vars.ps1")
        
    # Everything set up correctly
    elif gpu_hardware['has_amd'] and not pytorch_info['is_cpu_only'] and pytorch_info['rocm_available'] and env_vars['is_configured']:
        print(colored("✓ GOOD NEWS: AMD GPU is properly configured with PyTorch ROCm/HIP support", 'green'))
        print("\nYour system is correctly set up to use the AMD GPU for PyTorch acceleration.")
        print("You can verify performance with:")
        print("python test_amd_gpu_performance.py")
        
    # No AMD GPU
    elif not gpu_hardware['has_amd']:
        if gpu_hardware['has_nvidia'] and not pytorch_info['cuda_available']:
            print(colored("⚠️ ISSUE DETECTED: NVIDIA GPU present but CUDA not available in PyTorch", 'yellow'))
            print("\nYour system has an NVIDIA GPU, but PyTorch doesn't have CUDA support enabled.")
        elif not gpu_hardware['has_nvidia']:
            print("No dedicated GPU detected on this system.")
            print("Running on CPU only.")
    
    print_section_header("Next Steps")
    
    if gpu_hardware['has_amd']:
        if pytorch_info['is_cpu_only'] or not pytorch_info['rocm_available'] or not env_vars['is_configured']:
            print("1. Run one of the following scripts to set up AMD GPU support:")
            print("   - PowerShell: .\\install_amd_pytorch.ps1")
            print("   - Batch: install_amd_pytorch.bat")
            print("2. Set permanent environment variables (run as Administrator):")
            print("   - PowerShell: .\\set_permanent_amd_vars.ps1")
            print("3. Test GPU performance:")
            print("   python test_amd_gpu_performance.py")
            print("4. Use the AMD GPU for training:")
            print("   python train_robot_rl_demo.py --steps 10000 --parallel 8 --use-amd")
        else:
            print("Your AMD GPU is correctly configured. You can:")
            print("1. Test GPU performance:")
            print("   python test_amd_gpu_performance.py")
            print("2. Use the AMD GPU for training:")
            print("   python train_robot_rl_demo.py --steps 10000 --parallel 8 --use-amd")
    else:
        print("Refer to the README_DEMO.md for more information on running the robot positioning demo.")

def main():
    """Main function to run diagnostics."""
    print(colored("AMD GPU Compatibility Check for PyTorch", 'bold'))
    print("This script will diagnose your AMD GPU configuration for PyTorch.")
    
    # Get information about the environment
    pytorch_info = get_pytorch_info()
    gpu_hardware = check_gpu_hardware()
    env_vars = check_environment_variables()
    
    # Print diagnostic results
    print_diagnostic_results(pytorch_info, gpu_hardware, env_vars)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 