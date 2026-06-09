import numpy as np
import os
from tqdm import tqdm
from utils.follow_dot import label_pipeline, get_all_trajectories, store_coordinates
from utils.clean_dots import clean_coordinates, keep_full_trajectories


if __name__ == "__main__":
    # Count total frames in data/frames directory
    frames_dir = "data/frames"
    total_frames = len([f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))])
    print(f"Found {total_frames} frames in {frames_dir}")
    
    # First clean the coordinates to remove duplicates
    print("\nCleaning coordinates to remove duplicates (veery rare)...")
    min_dots = float('inf')
    for frame_number in tqdm(range(1, total_frames + 1), desc="Cleaning coordinates"):
        coords, removed = clean_coordinates(frame_number)
        min_dots = min(min_dots, len(coords))
        if removed > 0:
            print(f"Frame {frame_number}: removed {removed} points")
    
    print(f"Minimum dots across all frames: {min_dots}")
    num_frames = total_frames
    
    # Run the labeling pipeline with all frames
    if not (all(os.path.exists(f"data/labels_tracked/labels_tracked_{i}.npy") 
              for i in range(1, total_frames + 1)) and 
            all(os.path.exists(f"data/labels_raw/labels_raw_{i}.npy") 
              for i in range(1, total_frames + 1))):
        print("\nRunning complete labeling pipeline for all frames...")
        label_pipeline(total_frames)
    else:
        print("\nUsing existing label files")
    
    # Now compute trajectories for all frames
    print(f"\nComputing trajectories for all {num_frames} frames...")
    trajectories, is_new = get_all_trajectories(min_dots, num_frames)
    
    # Store coordinates for the computed trajectories
    print("\nStoring coordinates...")
    store_coordinates(min_dots)
    
    # Clean trajectories to keep only dots present in all frames
    print("\nCleaning trajectories...")
    keep_full_trajectories("dot_trajectories.json")
    
    # Print a few trajectories to check they make sense
    print("\nChecking first few trajectories:")
    for dot_idx in range(5):  # Check first 5 dots
        try:
            traj = np.load(f"data/trajectories/trajectory_{dot_idx}.npy")
            print(f"\nDot {dot_idx} trajectory:")
            print(f"Shape: {traj.shape}")
            print(f"First 5 positions: \n{traj[:5]}")  # Show just first 5 positions
            print(f"Last 5 positions: \n{traj[-5:]}")  # Show last 5 positions
        except Exception as e:
            print(f"Could not load trajectory for dot {dot_idx}: {e}")