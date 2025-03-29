#!/usr/bin/env python3
"""
Performance monitoring script for training.
This checks GPU utilization and training speed.
"""

import os
import time
import psutil  # type: ignore
import subprocess
from datetime import datetime

def get_gpu_info():
    """Get GPU information using Windows wmic command"""
    try:
        # GPU utilization can't be directly queried in Windows without additional tools
        # We'll just get the name for now
        output = subprocess.check_output(
            "wmic path win32_VideoController get Name", 
            shell=True
        ).decode("utf-8").strip()
        return output
    except Exception as e:
        return f"Error getting GPU info: {e}"

def get_process_info(name="python"):
    """Get information about Python processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time']):
        try:
            # Check if it's a Python process
            if name.lower() in proc.info['name'].lower():
                # Get additional info
                with proc.oneshot():
                    memory_mb = proc.memory_info().rss / (1024 * 1024)
                    create_time = datetime.fromtimestamp(proc.create_time()).strftime('%H:%M:%S')
                    cmdline = ' '.join(proc.cmdline())
                    
                    # Add to process list
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': memory_mb,
                        'create_time': create_time,
                        'cmdline': cmdline[:50] + ('...' if len(cmdline) > 50 else '')
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def check_model_growth(model_dir):
    """Check model directory for checkpoint files and their sizes"""
    if not os.path.exists(model_dir):
        return "Model directory not found"
    
    # Get checkpoint files
    files = []
    total_size_mb = 0
    
    for filename in os.listdir(model_dir):
        if filename.startswith("checkpoint") or filename == "final_model":
            filepath = os.path.join(model_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            files.append((filename, size_mb))
            total_size_mb += size_mb
    
    # Sort by name
    files.sort()
    
    result = f"Model files: {len(files)}, Total size: {total_size_mb:.2f} MB\n"
    for name, size in files:
        result += f"  {name}: {size:.2f} MB\n"
    
    return result

def main():
    """Monitor performance continuously"""
    model_dir = "models/ppo_optimized"  # Default model directory
    
    # Allow specification of model directory via command line
    if len(os.sys.argv) > 1:
        model_dir = os.sys.argv[1]
    
    print(f"Monitoring performance for model directory: {model_dir}")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 80)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\nIteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 40)
            
            # System overview
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            print(f"CPU usage: {cpu_percent}%")
            print(f"Memory: {memory.percent}% (Used: {memory.used / (1024**3):.1f} GB, Available: {memory.available / (1024**3):.1f} GB)")
            
            # GPU info
            print("\nGPU Information:")
            print(get_gpu_info())
            
            # Python processes
            print("\nPython Processes:")
            processes = get_process_info()
            for proc in processes:
                print(f"PID: {proc['pid']}, CPU: {proc['cpu_percent']}%, RAM: {proc['memory_mb']:.1f} MB, Started: {proc['create_time']}")
                print(f"  {proc['cmdline']}")
            
            # Model directory info
            print("\nModel Information:")
            print(check_model_growth(model_dir))
            
            # Wait before next check
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main() 