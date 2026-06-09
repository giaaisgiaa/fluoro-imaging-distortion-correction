## Stage 2: Dot Follow

This second stage is a custom dot tracking system that follows individual dots across multiple frames using a Nearest Neighbor Search (NNS) approach. The system is designed to be memory-efficient and handle large numbers of dots across multiple frames.

### System Overview

The tracking system consists of several key components:

1. **Coordinate Cleaning** - Removes duplicate dots (rare occurrences). 
                          U-Net showed only in the very first frame (reference) 2 double dot detections, 
                          very close to each other
2. **Grid Labeling** - Labels each dot with a unique identifier
3. **Trajectory Tracking** - Follows dots across frames using custom nearest neighbor search algorithm and 
                          updates the dots indexes accordingly 
4. **Coordinate Storage** - Saves tracked trajectories in both NumPy and JSON formats

### Directory Structure

```
Stage 2/
├── data/
│   ├── coordinates/          # JSON coordinate files
│   ├── coordinates_numpy/    # NumPy coordinate files
│   ├── frames/              # Input frame images
│   ├── labels_raw/          # Initial dot labels
│   ├── labels_tracked/      # Tracked dot labels
│   └── trajectories/        # Final dot trajectories
├── utils/
│   ├── clean_dots.py        # Coordinate cleaning utilities
│   ├── data_loading.py      # Data loading functions
│   ├── follow_dot.py        # Core tracking implementation
│   └── visualization.py     # Visualization utilities
├── follow_dot.py            # Main execution script
└── dot_trajectories.json    # Final trajectory data
```

### Using the Scripts

#### 1. Dot Tracking (`follow_dot.py`)

This is the main script that processes the frames and tracks dots:

```bash
python follow_dot.py
```

The script will:
1. Count total frames in `data/frames/`
2. Clean coordinates to remove any duplicates
3. Run the labeling pipeline if needed
4. Compute and save trajectories
5. Generate `dot_trajectories.json`

Output files will be created in their respective directories under `data/`.

#### 2. Visualization (`visualize_trajectories.py`)

This script provides interactive visualization of the tracked dots:

```bash
python visualize_trajectories.py
```

The script offers two visualization modes:

1. **All Trajectories Mode** (Option 1):
   - Shows all dot trajectories overlaid on a single plot
   - Creates `all_trajectories.png`
   - Useful for overall movement pattern analysis
   - Shows start points (green) and end points (red)

2. **Selected Dots Mode** (Option 2):
   - Allows detailed analysis of specific dots
   - Interactive prompts for:
     ```
     Enter dot indices to track (space-separated numbers, e.g., '1 2 3'): 
     Enter number of frames to process (press Enter for default=150):
     Generate animation? (y/n, press Enter for yes):
     ```
   - Outputs:
     - Individual trajectory plots: `dot_X_trajectory.png`
     - Optional animation: `dots_X_Y_Z_animation.mp4`

### Output Files

After running both scripts, you'll have:

1. **Tracking Data**:
   - `data/labels_tracked/`: Tracked dot positions
   - `data/trajectories/`: Individual dot trajectories
   - `dot_trajectories.json`: Complete tracking data

2. **Visualizations**:
   - `all_trajectories.png`: Overview of all dot movements
   - `dot_X_trajectory.png`: Individual dot trajectory plots
   - `dots_X_Y_Z_animation.mp4`: Animated visualization (if enabled)

### Notes

- Dot indices are 1-based (as stored in labels)
- The visualization script requires the tracking script to be run first
- Animations require ffmpeg to be installed
- For large datasets, generating all trajectories visualization might take longer