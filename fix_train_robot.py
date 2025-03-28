#!/usr/bin/env python3
# fix_train_robot.py
# Script to fix issues in the train_robot_rl_positioning_revamped.py file

import os
import re
import shutil
import sys

def fix_file(filename):
    print(f"Reading {filename}...")
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a backup
    backup_file = f"{filename}.bak"
    print(f"Creating backup at {backup_file}")
    shutil.copy2(filename, backup_file)
    
    # Fix 1: Remove strict_limits from create_revamped_envs function signature
    print("Fixing create_revamped_envs function signature...")
    content = re.sub(
        r'def create_revamped_envs\(num_envs=1, viz_speed=0\.0, parallel_viz=False, strict_limits=False, training_mode=True\):',
        'def create_revamped_envs(num_envs=1, viz_speed=0.0, parallel_viz=False, training_mode=True):',
        content
    )
    
    # Fix 2: Remove wrapping in JointLimitEnforcingEnv in create_revamped_envs
    print("Removing JointLimitEnforcingEnv wrapping...")
    content = re.sub(
        r'# Wrap in JointLimitEnforcingEnv if requested\s+if strict_limits:\s+env = JointLimitEnforcingEnv\(env\)',
        '',
        content
    )
    
    # Fix 3: Remove strict_limits arg in calls to create_revamped_envs
    print("Fixing create_revamped_envs function calls...")
    content = re.sub(
        r'create_revamped_envs\(\s*num_envs=args\.parallel,\s*viz_speed=args\.viz_speed[^,]+,\s*parallel_viz=[^,]+,(?:\s*strict_limits=[^,]+,)?\s*training_mode=True\s*\)',
        'create_revamped_envs(\n        num_envs=args.parallel,\n        viz_speed=args.viz_speed if args.gui else 0.0,\n        parallel_viz=args.parallel_viz and args.gui,\n        training_mode=True\n    )',
        content
    )
    
    # Fix 4: Remove strict_limits parameter from parse_args function
    print("Fixing parse_args function...")
    content = re.sub(
        r'parser\.add_argument\(\'--strict-limits\', action=\'store_true\', help=\'[^\']+\'\)',
        '',
        content
    )
    
    # Fix 5: Update imageio import in run_evaluation_sequence
    print("Fixing imageio handling...")
    
    # Fix video saving at the end
    content = re.sub(
        r'# Save video if requested\s+if save_video:\s+from datetime import datetime\s+video_path = f"\./evaluation_{datetime\.now\(\)\.strftime\(\'%Y%m%d_%H%M%S\'\)}\.mp4"\s+imageio\.mimsave\(video_path, frames, fps=30\)\s+print\(f"Video saved to {video_path}"\)',
        '# Save video if requested\n    if save_video:\n        try:\n            from datetime import datetime\n            video_path = f"./evaluation_{datetime.now().strftime(\'%Y%m%d_%H%M%S\')}.mp4"\n            imageio.mimsave(video_path, frames, fps=30)\n            print(f"Video saved to {video_path}")\n        except Exception as e:\n            print(f"Error saving video: {e}")',
        content
    )
    
    # Fix 6: Remove strict_limits from evaluate_model and run_evaluation_sequence calls
    print("Fixing evaluate_model and run_evaluation_sequence calls...")
    content = re.sub(
        r'evaluate_model\(\s*model_path=args\.load,\s*num_episodes=args\.eval_episodes,\s*visualize=args\.gui,\s*verbose=args\.verbose(?:,\s*strict_limits=args\.strict_limits)?\s*\)',
        'evaluate_model(\n                model_path=args.load,\n                num_episodes=args.eval_episodes,\n                visualize=args.gui,\n                verbose=args.verbose\n            )',
        content
    )
    
    content = re.sub(
        r'run_evaluation_sequence\(\s*model_path=args\.load,\s*viz_speed=[^,]+,\s*save_video=args\.save_video(?:,\s*strict_limits=args\.strict_limits)?\s*\)',
        'run_evaluation_sequence(\n                model_path=args.load,\n                viz_speed=args.viz_speed if args.viz_speed > 0 else 0.02,\n                save_video=args.save_video\n            )',
        content
    )
    
    # Write the fixed content back to the file
    print(f"Writing fixed content back to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixes completed!")

if __name__ == "__main__":
    target_file = "res/rml/python/train_robot_rl_positioning_revamped.py"
    
    if not os.path.exists(target_file):
        print(f"Error: File {target_file} not found.")
        sys.exit(1)
    
    fix_file(target_file) 