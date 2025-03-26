#!/usr/bin/env python3
# Script to check indentation in the file

with open('train_robot_rl_ppo_directml.py', 'r') as f:
    lines = f.readlines()

# Open a file to write the results
with open('indentation_check.txt', 'w') as out:
    # Print the lines around the reported error
    for i in range(75, 90):
        line_num = i + 1
        line = lines[i].rstrip('\n')
        out.write(f"{line_num:3d}: {repr(line)}\n")

print("Results written to indentation_check.txt") 