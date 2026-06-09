import numpy as np
import os
import json
from tqdm import tqdm

from utils.data_loading import load_grid_centers


def label_grid(frame_nr, points):
    """Labels each pixel position with dot indices."""
    label_file_path = f"data/labels_raw/labels_raw_{frame_nr}.npy"  # Changed to .npy
    if os.path.exists(label_file_path):
        return None, np.array([])
    
    # Create a label matrix the size of the image (width=1664, height=1600)
    labels = np.zeros((1600, 1664), dtype=int)  # Shape is (height, width)
    counter = 0
    
    # Directly assign dot indices to their pixel positions
    for idx, (x, y) in enumerate(points):
        # Coordinates are already integers
        x, y = int(x), int(y)
        
        # Check if coordinates are within image bounds (x < width, y < height)
        if 0 <= x < 1664 and 0 <= y < 1600:  # width=1664, height=1600
            if labels[y, x] == 0:  # Only count if we're not overwriting an existing label
                labels[y, x] = idx + 1  # Add 1 to avoid using 0 as a label
                counter += 1
    
    print(f"Frame {frame_nr}: Labeled {counter} dots")
    np.save(label_file_path, labels)  # Changed to np.save
    return labels, np.array([])


def search_correspondences(frame_nr, labels_prev, labels_curr, neighbor_range=5):
    """
    Search for each dot in a 5x5 pixel neighborhood between frames.
    More careful matching to avoid wrong assignments.
    """
    tracked_file_path = f"data/labels_tracked/labels_tracked_{frame_nr}.npy"
    if os.path.exists(tracked_file_path):
        labels = np.load(tracked_file_path)
        count = np.count_nonzero(labels)
        return labels, count
    
    if frame_nr == 1:
        np.save(tracked_file_path, labels_curr)
        count = np.count_nonzero(labels_curr)
        return labels_curr, count
    
    height, width = labels_prev.shape  # height=1600, width=1664
    labels = np.zeros((height, width), dtype=int)
    count = 0
    
    # Find all dots in previous frame
    prev_dots = {}  # Store coordinates for each dot index
    for y in range(height):
        for x in range(width):
            if labels_prev[y, x] != 0:
                prev_dots[labels_prev[y, x]] = (y, x)
    
    # For each dot in previous frame, find best match in current frame
    for dot_idx, (prev_y, prev_x) in prev_dots.items():
        y_start = max(prev_y - neighbor_range, 0)
        y_end = min(prev_y + neighbor_range, height)
        x_start = max(prev_x - neighbor_range, 0)
        x_end = min(prev_x + neighbor_range, width)
        
        # Find all candidate dots in neighborhood
        candidates = []
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if labels_curr[y, x] != 0 and labels[y, x] == 0:  # Only consider unassigned dots
                    # Calculate distance from previous position
                    dist = ((y - prev_y) ** 2 + (x - prev_x) ** 2) ** 0.5
                    candidates.append((dist, y, x))
        
        # If we found candidates, assign label to the closest one
        if candidates:
            # Sort by distance and take the closest
            candidates.sort()  # Sort by distance (first element of tuple)
            best_y, best_x = candidates[0][1:]  # Take coordinates of closest dot
            
            # Assign the label only if not already assigned
            if labels[best_y, best_x] == 0:
                labels[best_y, best_x] = dot_idx
                count += 1
    
    np.save(tracked_file_path, labels)
    return labels, count


def get_all_trajectories(num_dots, num_frames, labels_dir='data/labels_tracked'):
    """Gets trajectories for all dots across frames using only tracked labels."""
    if os.path.exists("data/trajectories/trajectories_full_grid.npy"):
        return np.load("data/trajectories/trajectories_full_grid.npy"), False
    
    os.makedirs("data/trajectories", exist_ok=True)
    
    trajectories = np.full((num_dots, num_frames, 2), np.nan)
    print("Processing frames...")
    
    # Pre-load all label files into memory
    print("Loading label files...")
    label_matrices = []
    for frame in tqdm(range(1, num_frames + 1), desc="Loading labels"):
        label_matrices.append(np.load(f"{labels_dir}/labels_tracked_{frame}.npy"))
    
    # Process all frames
    print("\nExtracting trajectories...")
    for frame_idx, tracked_labels in enumerate(tqdm(label_matrices, desc="Processing frames")):
        # Find all non-zero positions at once
        y_coords, x_coords = np.nonzero(tracked_labels)
        labels = tracked_labels[y_coords, x_coords]
        
        # Assign coordinates to trajectories
        for y, x, label in zip(y_coords, x_coords, labels):
            dot_index = label - 1  # Subtract 1 since we stored labels as idx + 1
            trajectories[dot_index, frame_idx] = (x, y)
    
    # Save full grid
    print("\nSaving trajectories...")
    np.save("data/trajectories/trajectories_full_grid.npy", trajectories)
    
    # Save individual trajectories as .npy files
    for dot_index in tqdm(range(num_dots), desc="Saving individual trajectories"):
        trajectory = trajectories[dot_index]
        valid_frames = ~np.isnan(trajectory[:, 0])
        if np.any(valid_frames):
            valid_trajectory = trajectory[valid_frames]
            np.save(f"data/trajectories/trajectory_{dot_index}.npy", valid_trajectory)
    
    return trajectories, True


def store_coordinates(num_dots):
    """
    Creates a JSON file with coordinates for each frame and dot.
    Structure: {frame_number: {dot_index: {"coordinates": [x, y]}}}
    """
    if os.path.exists("dot_trajectories.json"):
        with open("dot_trajectories.json", "r") as f:
            return json.load(f), False
    
    # Load the full trajectories matrix
    print("Loading trajectories...")
    all_trajectories = np.load("data/trajectories/trajectories_full_grid.npy")
    num_frames = all_trajectories.shape[1]
    print(f"Loaded trajectories with shape: {all_trajectories.shape}")
    
    coordinate_data = {}
    print(f"\nExtracting coordinates for {num_frames} frames...")
    
    # Process each frame
    for frame_number in tqdm(range(num_frames), desc="Processing frames"):
        coordinate_data[frame_number + 1] = {}
        
        # Process each dot
        for dot_index in range(num_dots):
            coordinates = all_trajectories[dot_index, frame_number]
            
            # Only store if coordinates are valid (not NaN)
            if not np.isnan(coordinates[0]):
                coordinate_data[frame_number + 1][dot_index] = {
                    "coordinates": [float(coordinates[0]), float(coordinates[1])]
                }
    
    formatted_data = {f"frame {k}": v for k, v in sorted(coordinate_data.items(), key=lambda x: int(x[0]))}
    
    print("\nSaving JSON file...")
    with open("dot_trajectories.json", "w") as f:
        json.dump(formatted_data, f, indent=4)
    
    return coordinate_data, True


def label_pipeline(num_frames):
    """Executes the complete labeling pipeline."""
    all_tracked_exist = all(os.path.exists(f"data/labels_tracked/labels_tracked_{i}.npy") 
                          for i in range(1, num_frames + 1))
    all_raw_exist = all(os.path.exists(f"data/labels_raw/labels_raw_{i}.npy")  
                       for i in range(1, num_frames + 1))
    
    if all_tracked_exist and all_raw_exist:
        return False
        
    # Label each frame
    print("Labeling frames with dot indices...")
    for n in tqdm(range(1, num_frames + 1), desc="Creating labels_raw"):
        points = load_grid_centers(n)
        label_grid(frame_nr=n, points=points)

    # Start tracking from frame 1
    print("\nTracking dots across frames...")
    labels_raw_1 = np.load("data/labels_raw/labels_raw_1.npy")  
    search_correspondences(frame_nr=1, labels_prev=labels_raw_1, labels_curr=labels_raw_1, neighbor_range=0)

    # Track dots across subsequent frames
    for n in tqdm(range(1, num_frames), desc="Creating labels_tracked"):
        frame_curr = n + 1
        
        # For frame 1, use raw labels from frame 1
        if n == 1:
            labels_ref = np.load("data/labels_raw/labels_raw_1.npy")  
        else:
            # For other frames, use previous frame's tracked labels
            labels_ref = np.load(f"data/labels_tracked/labels_tracked_{n}.npy")

        # Get current frame's raw labels
        labels_curr = np.load(f"data/labels_raw/labels_raw_{frame_curr}.npy")  
        
        # Track dots with fixed 5-pixel neighborhood
        labels, _ = search_correspondences(frame_curr, 
                                    labels_prev=labels_ref, 
                                    labels_curr=labels_curr, 
                                    neighbor_range=5)
        labels_ref = labels

    return True


def process_complete_pipeline(num_frames, num_dots):
    """Process the complete pipeline with file existence checks."""
    # Run labeling pipeline
    if label_pipeline(num_frames):
        print("Completed labeling pipeline")
    else:
        print("Using existing label files")

    # Get trajectories
    trajectories, is_new = get_all_trajectories(num_dots, num_frames)
    if is_new:
        print("Generated new trajectories")
    else:
        print("Using existing trajectories")

    # Store coordinates
    coordinate_data, is_new = store_coordinates(num_dots)
    if is_new:
        print("Generated new coordinate data")
    else:
        print("Using existing coordinate data")

    return True