# high_level_interface.py
import numpy as np
from stable_baselines3 import PPO
import pybullet as p
import tkinter as tk
from tkinter import ttk
import threading
import time

from robot_sim import FANUCRobotEnv

class RobotControlInterface:
    def __init__(self, model_path="./models/fanuc_final_model.zip"):
        # Load the trained RL model
        self.model = PPO.load(model_path)
        
        # Create the robot environment
        self.env = FANUCRobotEnv(render=True)
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("FANUC LR Mate 200iC Control Interface")
        self.root.geometry("800x600")
        
        # Create tabs
        self.tabControl = ttk.Notebook(self.root)
        self.tab_direct = ttk.Frame(self.tabControl)
        self.tab_task = ttk.Frame(self.tabControl)
        
        self.tabControl.add(self.tab_direct, text="Direct Control")
        self.tabControl.add(self.tab_task, text="Task Control")
        self.tabControl.pack(expand=1, fill="both")
        
        # Direct Control Tab
        self.setup_direct_control_tab()
        
        # Task Control Tab
        self.setup_task_control_tab()
        
        # Status Frame
        self.status_frame = ttk.LabelFrame(self.root, text="Status")
        self.status_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(padx=5, pady=5)
        
        # Control Thread
        self.control_thread = None
        self.running = False
        
    def setup_direct_control_tab(self):
        # Joint sliders
        self.joint_frame = ttk.LabelFrame(self.tab_direct, text="Joint Control")
        self.joint_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.joint_sliders = []
        for i in range(6):
            frame = ttk.Frame(self.joint_frame)
            frame.pack(fill="x", padx=5, pady=5)
            
            label = ttk.Label(frame, text=f"Joint {i+1}")
            label.pack(side="left", padx=5)
            
            low, high = np.rad2deg(self.env.joint_limits[i])
            slider = ttk.Scale(frame, from_=low, to=high, orient="horizontal", length=300)
            slider.set(0)  # Set initial value to 0
            slider.pack(side="right", fill="x", expand=True, padx=5)
            
            self.joint_sliders.append(slider)
        
        # Control buttons
        control_frame = ttk.Frame(self.tab_direct)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.home_button = ttk.Button(control_frame, text="Home Position", command=self.home_robot)
        self.home_button.pack(side="left", padx=5)
        
        self.send_button = ttk.Button(control_frame, text="Send to Robot", command=self.send_joint_positions)
        self.send_button.pack(side="right", padx=5)
        
    def setup_task_control_tab(self):
        # Task control options
        self.task_frame = ttk.LabelFrame(self.tab_task, text="Task Specification")
        self.task_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Position control
        pos_frame = ttk.Frame(self.task_frame)
        pos_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(pos_frame, text="Target Position:").grid(row=0, column=0, padx=5, pady=5)
        
        x_frame = ttk.Frame(pos_frame)
        x_frame.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(x_frame, text="X:").pack(side="left")
        self.x_entry = ttk.Entry(x_frame, width=10)
        self.x_entry.insert(0, "0.5")
        self.x_entry.pack(side="left")
        
        y_frame = ttk.Frame(pos_frame)
        y_frame.grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(y_frame, text="Y:").pack(side="left")
        self.y_entry = ttk.Entry(y_frame, width=10)
        self.y_entry.insert(0, "0.0")
        self.y_entry.pack(side="left")
        
        z_frame = ttk.Frame(pos_frame)
        z_frame.grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(z_frame, text="Z:").pack(side="left")
        self.z_entry = ttk.Entry(z_frame, width=10)
        self.z_entry.insert(0, "0.5")
        self.z_entry.pack(side="left")
        
        # Task type
        task_type_frame = ttk.Frame(self.task_frame)
        task_type_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(task_type_frame, text="Task Type:").pack(side="left", padx=5)
        
        self.task_type = tk.StringVar()
        task_combobox = ttk.Combobox(task_type_frame, textvariable=self.task_type)
        task_combobox['values'] = ('Move To Position', 'Pick and Place', 'Follow Trajectory')
        task_combobox.current(0)
        task_combobox.pack(side="left", padx=5)
        
        # Execute button
        self.execute_button = ttk.Button(self.task_frame, text="Execute Task", command=self.execute_task)
        self.execute_button.pack(pady=10)
        
    def home_robot(self):
        """Set all joint sliders to 0"""
        for slider in self.joint_sliders:
            slider.set(0)
        
        self.send_joint_positions()
    
    def send_joint_positions(self):
        """Send the current joint positions to the robot"""
        joint_positions = [np.deg2rad(slider.get()) for slider in self.joint_sliders]
        
        # Update status
        self.status_label.config(text="Moving to joint positions...")
        
        # Execute in a separate thread to avoid blocking the GUI
        if self.control_thread is not None and self.control_thread.is_alive():
            self.running = False
            self.control_thread.join()
        
        self.running = True
        self.control_thread = threading.Thread(target=self._move_to_joint_positions, args=(joint_positions,))
        self.control_thread.start()
    
    def _move_to_joint_positions(self, joint_positions):
        """Thread function to move the robot to specified joint positions"""
        self.env.step(joint_positions)
        time.sleep(0.5)  # Give time for the robot to move
        
        # Update status when done
        self.status_label.config(text="Joint position command completed")
        self.running = False
    
    def execute_task(self):
        """Execute the selected task using the RL model"""
        task_type = self.task_type.get()
        
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            z = float(self.z_entry.get())
        except ValueError:
            self.status_label.config(text="Error: Position values must be numbers")
            return
        
        target_position = np.array([x, y, z])
        
        # Update status
        self.status_label.config(text=f"Executing task: {task_type}...")
        
        # Execute in a separate thread
        if self.control_thread is not None and self.control_thread.is_alive():
            self.running = False
            self.control_thread.join()
        
        self.running = True
        self.control_thread = threading.Thread(target=self._execute_task_thread, args=(task_type, target_position))
        self.control_thread.start()
    
    def _execute_task_thread(self, task_type, target_position):
        """Thread function to execute the selected task"""
        # Reset the environment
        obs = self.env.reset()
        
        # Set the target position
        self.env.target_position = target_position
        
        # Use the RL model to plan and execute the movement
        done = False
        steps = 0
        max_steps = 100  # Prevent infinite loops
        
        while not done and steps < max_steps and self.running:
            # Get action from the model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute the action
            obs, reward, done, info = self.env.step(action)
            
            # Short delay for visualization
            time.sleep(0.05)
            
            steps += 1
            
            # Update status with distance to target
            distance = info.get('distance', float('inf'))
            self.status_label.config(text=f"Executing task: {task_type}... Distance: {distance:.4f}")
        
        if done:
            self.status_label.config(text=f"Task {task_type} completed successfully")
        elif steps >= max_steps:
            self.status_label.config(text=f"Task {task_type} timed out")
        else:
            self.status_label.config(text=f"Task {task_type} interrupted")
        
        self.running = False
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()
        
    def close(self):
        """Clean up resources"""
        if self.control_thread is not None and self.control_thread.is_alive():
            self.running = False
            self.control_thread.join()
            
        self.env.close()

if __name__ == "__main__":
    # Check if the model exists, if not, inform the user
    model_path = "./models/fanuc_final_model.zip"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train_rl.py")
        exit(1)
        
    # Create and run the interface
    interface = RobotControlInterface(model_path)
    try:
        interface.run()
    finally:
        interface.close()