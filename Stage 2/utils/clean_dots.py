import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def clean_coordinates(frame_number, neighbor_range=7):
    """
    Clean coordinates by removing one of two points that are too close to each other. 
    Until now only the first frame showed 2 dots (bottom left) which were detected twice. 
    
    Args:
        frame_number (int): Frame number to process
        neighbor_range (int): Distance range in pixels under 
        which only one of the two dots is kept.
    
    Returns:
        list: Cleaned coordinates
        int: Number of points removed
    """
    # Load coordinates
    coord_file = f"data/coordinates/grid_centers_{frame_number}.json"
    with open(coord_file, 'r') as f:
        coordinates = json.load(f)
    
    original_count = len(coordinates)
    points_to_keep = np.ones(len(coordinates), dtype=bool)
    
    # Compare each point with every other point
    for i in range(len(coordinates)):
        if not points_to_keep[i]:
            continue
            
        point1 = coordinates[i]
        x1, y1 = point1['x'], point1['y']
        
        for j in range(i + 1, len(coordinates)):
            if not points_to_keep[j]:
                continue
                
            point2 = coordinates[j]
            x2, y2 = point2['x'], point2['y']
            
            # Check if points are too close in both x and y directions
            if (abs(x1 - x2) < neighbor_range and 
                abs(y1 - y2) < neighbor_range):
                # Randomly choose which point to remove
                to_remove = np.random.choice([i, j])
                points_to_keep[to_remove] = False
    
    # Keep only the selected points
    cleaned_coordinates = [coord for i, coord in enumerate(coordinates) if points_to_keep[i]]
    points_removed = original_count - len(cleaned_coordinates)
    
    # Save cleaned coordinates back to file
    with open(coord_file, 'w') as f:
        json.dump(cleaned_coordinates, f, indent=2)
    
    # Visualize the cleaning result if points were removed
    if points_removed > 0 and frame_number == 1:  # Only visualize frame 1 if it was cleaned
        frame = cv2.imread(f"data/frames/frame_{frame_number}.png")
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create plot with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Original coordinates
            ax1.imshow(frame)
            points = np.array([(c['x'], c['y']) for c in coordinates])
            ax1.scatter(points[:, 0], points[:, 1], c='r', s=20, alpha=0.6, 
                       label=f'Original dots ({len(coordinates)} total)')
            ax1.set_title(f'Frame {frame_number} - Original Coordinates')
            ax1.legend()
            
            # Cleaned coordinates
            ax2.imshow(frame)
            cleaned_points = np.array([(c['x'], c['y']) for c in cleaned_coordinates])
            ax2.scatter(cleaned_points[:, 0], cleaned_points[:, 1], c='g', s=20, alpha=0.6,
                       label=f'Cleaned dots ({len(cleaned_coordinates)} total)')
            ax2.set_title(f'Frame {frame_number} - Cleaned Coordinates\n({points_removed} dots removed)')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
    
    return cleaned_coordinates, points_removed


def keep_full_trajectories(input_file):
    """
    Clean trajectory data by keeping only dot indices that appear in all frames.
    Overwrites the original file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input JSON file containing trajectory data
    
    Returns:
    --------
    str : Path to the cleaned output file (same as input)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Cleaning trajectory data from {Path(input_file).name}")
    
    # Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get all frames
    frames = sorted(data.keys(), key=lambda k: int(k.split()[-1]))
    
    # Find common dot indices across all frames
    all_indices = set(data[frames[0]].keys())  # Start with indices from first frame
    for frame in frames[1:]:
        frame_indices = set(data[frame].keys())
        all_indices &= frame_indices  # Intersection
    
    all_indices = sorted([int(idx) for idx in all_indices])
    print(f"Found {len(all_indices)} dots present in all {len(frames)} frames")
    
    # Create new dataset with only consistent dots
    new_data = {}
    for frame in frames:
        new_data[frame] = {str(idx): data[frame][str(idx)] for idx in all_indices}
    
    # Overwrite the original file with nice formatting
    with open(input_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print(f"Overwritten original file: {input_file}")
    return input_file