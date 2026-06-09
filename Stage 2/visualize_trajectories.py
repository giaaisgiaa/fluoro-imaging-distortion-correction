"""
How to run this script:
--------------------------------------------------------------------------------
This script visualizes dot trajectories across frames interactively.

Usage:
    python visualize_trajectories.py

The script will guide you through two visualization options:

1. All trajectories mode:
   - Shows all dot trajectories overlaid on a single plot
   - Output: all_trajectories.png

2. Selected dots mode:
   - Shows detailed trajectories for dots you choose
   - You can customize:
     * Which dots to track (enter space-separated numbers, e.g., "1 2 3")
     * Number of frames to process (default: 150)
     * Whether to generate animation (y/n)
   - Output:
     * dot_X_trajectory.png : Individual trajectory plot for each selected dot
     * dots_X_Y_Z_animation.mp4 : Animation of selected dots' movement (if enabled)

Note: Dot indices are 1-based (as stored in labels)
--------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import json
import argparse
from tqdm import tqdm
import cv2
import os

def plot_simple_trajectory(dot_idx, trajectories, num_frames):
    """Create a simple 2D plot of a single dot trajectory"""
    x_coords = trajectories[dot_idx]['x']
    y_coords = trajectories[dot_idx]['y']
    frames = trajectories[dot_idx]['frames']
    
    if x_coords and y_coords:
        plt.figure(figsize=(12, 8))
        
        # Plot trajectory
        plt.plot(x_coords, y_coords, '-', color='blue', label='Trajectory', linewidth=2)
        
        # Plot start and end points
        plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End')
        
        # Add frame numbers at intervals
        num_annotations = 10
        if len(frames) > num_annotations:
            step = len(frames) // num_annotations
            for i in range(0, len(frames), step):
                plt.annotate(f'Frame {frames[i]}', 
                           (x_coords[i], y_coords[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
        
        plt.title(f'Trajectory of Dot {dot_idx} Across {num_frames} Frames')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig(f'dot_{dot_idx}_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_selected_dots(dot_indices, num_frames=150, save_animation=True):
    """Plot detailed trajectories for selected dots with optional animation"""
    if isinstance(dot_indices, int):
        dot_indices = [dot_indices]
    
    # Lists to store coordinates for each dot
    trajectories = {idx: {'x': [], 'y': [], 'frames': []} for idx in dot_indices}
    
    # Colors for different dots
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dot_indices)))
    
    # Load each frame and find the dots' positions
    print(f"Loading coordinates for dots {dot_indices}...")
    for frame in tqdm(range(1, num_frames + 1)):
        # Load the tracked labels for this frame
        labels = np.load(f"data/labels_tracked/labels_tracked_{frame}.npy")
        
        # Find positions for each dot
        for dot_idx in dot_indices:
            positions = np.where(labels == dot_idx)
            if len(positions[0]) > 0:
                y, x = positions[0][0], positions[1][0]
                trajectories[dot_idx]['x'].append(x)
                trajectories[dot_idx]['y'].append(y)
                trajectories[dot_idx]['frames'].append(frame)
    
    # Create simple trajectory plot for each dot
    for dot_idx in dot_indices:
        plot_simple_trajectory(dot_idx, trajectories, num_frames)
    
    if save_animation:
        print("\nCreating animation...")
        # Create figure for animation
        fig_anim, ax_anim = plt.subplots(figsize=(15, 10))
        
        def animate(frame_num):
            ax_anim.clear()
            
            # Load and display frame
            frame_img = cv2.imread(f"data/frames/frame_{frame_num+1}.png", cv2.IMREAD_GRAYSCALE)
            ax_anim.imshow(frame_img, cmap='gray')
            
            # Plot trajectories up to current frame
            for dot_idx, color in zip(dot_indices, colors):
                x_coords = trajectories[dot_idx]['x'][:frame_num+1]
                y_coords = trajectories[dot_idx]['y'][:frame_num+1]
                
                if x_coords and y_coords:
                    # Plot trajectory
                    ax_anim.plot(x_coords, y_coords, '-', color=color, 
                               label=f'Dot {dot_idx}', alpha=0.6, linewidth=2)
                    
                    # Plot current position
                    if frame_num < len(trajectories[dot_idx]['x']):
                        ax_anim.add_patch(Circle((x_coords[-1], y_coords[-1]), 
                                               radius=5, color=color, fill=True))
            
            ax_anim.set_title(f'Frame {frame_num+1}')
            ax_anim.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Create animation
        anim = animation.FuncAnimation(fig_anim, animate, frames=num_frames,
                                     interval=100, blit=False)
        
        # Save animation
        anim.save(f'dots_{"_".join(map(str, dot_indices))}_animation.mp4',
                 writer='ffmpeg', fps=10)
        
        plt.close(fig_anim)

def plot_all_trajectories():
    """Plot all trajectories in a single figure"""
    # Load trajectories
    with open("dot_trajectories.json", "r") as f:
        data = json.load(f)
    
    # Get number of frames and dots
    frames = sorted(data.keys(), key=lambda x: int(x.split()[1]))
    num_frames = len(frames)
    
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Get all unique dot indices
    all_dots = set()
    for frame in frames:
        all_dots.update(data[frame].keys())
    all_dots = sorted(map(int, all_dots))
    num_dots = len(all_dots)
    
    # Debug info
    print(f"Found {num_dots} unique dots")
    print(f"Number of frames: {num_frames}")
    
    # Count dots with trajectories
    dots_with_trajectories = 0
    missing_dots = []
    stationary_dots = []  # Dots that don't move
    
    # Plot trajectories
    print(f"\nAnalyzing trajectories...")
    for dot_idx in all_dots:
        # Collect coordinates for this dot across all frames
        x_coords = []
        y_coords = []
        frame_nums = []
        dot_str = str(dot_idx)
        
        # Check dot presence in frames
        frames_present = 0
        for frame in frames:
            if dot_str in data[frame]:
                frames_present += 1
                coords = data[frame][dot_str]["coordinates"]
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                frame_nums.append(int(frame.split()[1]))
        
        if x_coords:  # Only plot if we have coordinates
            dots_with_trajectories += 1
            
            # Check if dot is stationary
            if len(x_coords) > 1:
                start_pos = (x_coords[0], y_coords[0])
                end_pos = (x_coords[-1], y_coords[-1])
                if start_pos == end_pos:
                    all_same = all(x == x_coords[0] for x in x_coords) and all(y == y_coords[0] for y in y_coords)
                    if all_same:
                        stationary_dots.append((dot_idx, start_pos))
            
            # Plot trajectory
            plt.plot(x_coords, y_coords, '-', color='blue', alpha=0.5, linewidth=0.5)
            plt.scatter(x_coords[0], y_coords[0], color='green', s=20, alpha=0.3)
            plt.scatter(x_coords[-1], y_coords[-1], color='red', s=20, alpha=0.3)
            
            # Add frame numbers for some trajectories
            if dot_idx % 200 == 0:  # Label fewer dots to avoid clutter
                num_annotations = 5
                if len(frame_nums) > num_annotations:
                    step = len(frame_nums) // num_annotations
                    for i in range(0, len(frame_nums), step):
                        plt.annotate(f'Frame {frame_nums[i]}', 
                                   (x_coords[i], y_coords[i]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.7)
        else:
            missing_dots.append(dot_idx)
    
    print(f"\nActually plotted {dots_with_trajectories} trajectories")
    if missing_dots:
        print(f"Missing trajectories for {len(missing_dots)} dots")
    print(f"Found {len(stationary_dots)} stationary dots")
    
    plt.title(f'All Dot Trajectories Across {num_frames} Frames')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.plot([], [], color='blue', label='Trajectories')
    plt.scatter([], [], color='green', s=30, alpha=0.5, label='Start Points')
    plt.scatter([], [], color='red', s=30, alpha=0.5, label='End Points')
    plt.legend()
    
    # Save high-resolution figure
    plt.savefig('all_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("\nDot Trajectory Visualization")
    print("---------------------------")
    print("1. View all dot trajectories in a single plot")
    print("2. View specific dots with detailed trajectories and animation")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        print("\nGenerating visualization for all trajectories...")
        plot_all_trajectories()
    
    else:
        while True:
            dots_input = input("\nEnter dot indices to track (space-separated numbers, e.g., '1 2 3'): ").strip()
            try:
                dots = [int(x) for x in dots_input.split()]
                if not dots:
                    print("Please enter at least one dot index.")
                    continue
                if any(x < 1 for x in dots):
                    print("Dot indices must be 1 or greater.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces.")
        
        frames = input("\nEnter number of frames to process (press Enter for default=150): ").strip()
        frames = int(frames) if frames else 150
        
        animation = input("\nGenerate animation? (y/n, press Enter for yes): ").strip().lower()
        animation = animation != 'n'
        
        print(f"\nGenerating visualization for dots {dots}...")
        plot_selected_dots(dots, frames, animation)