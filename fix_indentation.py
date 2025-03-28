#!/usr/bin/env python3
"""
Fix indentation issues in train_robot_rl_positioning_revamped.py
"""

import os
import re

def fix_indentation(input_file, output_file):
    """Fix specific indentation issues in the file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix issue around lines 2000-2010 (the else: clause)
    pattern1 = r'(\s+)else:\s*\n(\s+)plt\.plot\(actions, label="Action"\)'
    replacement1 = r'\1else:\n\1    plt.plot(actions, label="Action")'
    content = re.sub(pattern1, replacement1, content)
    
    # Fix issue around line 2143 (try-except block)
    pattern2 = r'(\s+)try:\s*\n(\s+)import imageio'
    replacement2 = r'\1try:\n\1    import imageio'
    content = re.sub(pattern2, replacement2, content)
    
    # Fix issue around line 2143 (try-except block lines afterward)
    pattern3 = r'(\s+)frames = \[\]\s*\n(\s+)# Set higher resolution for video'
    replacement3 = r'\1    frames = []\n\1    # Set higher resolution for video'
    content = re.sub(pattern3, replacement3, content)
    
    # Fix further indentation inside the try block
    pattern4 = r'(\s+)p\.configureDebugVisualizer\(p\.COV_ENABLE_GUI, 0\)'
    replacement4 = r'\1    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)'
    content = re.sub(pattern4, replacement4, content)
    
    pattern5 = r'(\s+)p\.resetDebugVisualizerCamera\('
    replacement5 = r'\1    p.resetDebugVisualizerCamera('
    content = re.sub(pattern5, replacement5, content)
    
    # Fix the camera parameters indentation
    pattern6 = r'(\s+)cameraDistance=1\.2,\s*\n(\s+)cameraYaw=120,'
    replacement6 = r'\1        cameraDistance=1.2,\n\1        cameraYaw=120,'
    content = re.sub(pattern6, replacement6, content)
    
    pattern7 = r'(\s+)cameraPitch=-20,\s*\n(\s+)cameraTargetPosition='
    replacement7 = r'\1        cameraPitch=-20,\n\1        cameraTargetPosition='
    content = re.sub(pattern7, replacement7, content)
    
    # Fix issue around line 2230 (another try-except block)
    pattern8 = r'(\s+)try:\s*\n(\s+)from datetime import datetime'
    replacement8 = r'\1try:\n\1    from datetime import datetime'
    content = re.sub(pattern8, replacement8, content)
    
    # Fix remaining lines in that try block
    pattern9 = r'(\s+)video_path = f"./evaluation'
    replacement9 = r'\1    video_path = f"./evaluation'
    content = re.sub(pattern9, replacement9, content)
    
    pattern10 = r'(\s+)imageio\.mimsave'
    replacement10 = r'\1    imageio.mimsave'
    content = re.sub(pattern10, replacement10, content)
    
    pattern11 = r'(\s+)print\(f"Video saved to'
    replacement11 = r'\1    print(f"Video saved to'
    content = re.sub(pattern11, replacement11, content)
    
    # Write the fixed content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed content written to {output_file}")

# Fix the indentation
input_file = 'src/core/train_robot_rl_positioning_revamped.py'
output_file = 'src/core/train_robot_rl_positioning_revamped_fixed.py'
fix_indentation(input_file, output_file) 