import json
import numpy as np
from pathlib import Path

def clean_trajectories(input_file):
    """
    Clean trajectory data by keeping only dot indices that appear in all frames.
    
    Parameters:
    -----------
    input_file : str
        Path to the input JSON file
    """
    print("="*60)
    print(f"CLEANING TRAJECTORIES FROM {Path(input_file).name}")
    print("="*60)
    
    # Load the data
    print("Loading data...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get all frames
    frames = sorted(data.keys(), key=lambda k: int(k.split()[-1]))
    print(f"Found {len(frames)} frames")
    
    # Find common dot indices across all frames
    print("\nAnalyzing dot indices...")
    all_indices = set(data[frames[0]].keys())  # Start with indices from first frame
    for frame in frames[1:]:
        frame_indices = set(data[frame].keys())
        all_indices &= frame_indices  # Intersection
    
    all_indices = sorted([int(idx) for idx in all_indices])
    print(f"Found {len(all_indices)} dots that appear in all frames")
    
    # Create new dataset with only consistent dots
    print("\nCreating cleaned dataset...")
    new_data = {}
    for frame in frames:
        new_data[frame] = {str(idx): data[frame][str(idx)] for idx in all_indices}
    
    # Save to new file
    output_file = str(Path(input_file).parent / f"{Path(input_file).stem}_cleaned.json")
    print(f"\nSaving cleaned data to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(new_data, f)
    
    # Verify the new file
    print("\nVerifying cleaned data:")
    with open(output_file, 'r') as f:
        verify_data = json.load(f)
    
    frames = sorted(verify_data.keys(), key=lambda k: int(k.split()[-1]))
    print("\nFirst 5 frames:")
    for frame in frames[:5]:
        print(f"Frame {frame}: {len(verify_data[frame])} dots")
    
    print("\nLast 5 frames:")
    for frame in frames[-5:]:
        print(f"Frame {frame}: {len(verify_data[frame])} dots")
    
    print("\nDone! Cleaned file created successfully.")
    return output_file

if __name__ == "__main__":
    print("="*60)
    print("TRAJECTORY CLEANER")
    print("="*60)
    
    # Find all JSON files in the data directory
    data_dir = Path("data")
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in the data directory!")
        exit(1)
    
    # Ask user to choose JSON file
    print("\nAvailable JSON files:")
    for i, file in enumerate(json_files, 1):
        print(f"{i}. {file.name}")
    
    while True:
        choice = input(f"\nEnter your choice (1-{len(json_files)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(json_files):
            break
        print(f"Invalid choice. Please enter a number between 1 and {len(json_files)}.")
    
    input_file = str(json_files[int(choice) - 1])
    clean_trajectories(input_file) 