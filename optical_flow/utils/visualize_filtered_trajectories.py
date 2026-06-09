import cv2
import numpy as np
import os
import glob
import json
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def preprocess_xray(image):
    """
    Preprocess X-ray image for better corner detection.
    """
    img_float = image.astype(np.float32)
    img_norm = cv2.normalize(img_float, None, 0, 1, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply((img_norm * 255).astype(np.uint8))
    img_denoised = cv2.fastNlMeansDenoising(img_clahe)
    return img_denoised

def detect_corners(image, max_corners=3000, quality_level=0.01, min_distance=10, is_preprocessed=False):
    """
    Detect corners using Shi-Tomasi corner detector.
    """
    if not is_preprocessed:
        processed = preprocess_xray(image)
    else:
        processed = image  # Already preprocessed
        quality_level = 0.2  # Higher quality threshold for binary images
        min_distance = 15   # Increased minimum distance for preprocessed dots
    
    corners = cv2.goodFeaturesToTrack(
        processed,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )
    return corners

def filter_corners_by_dot_centers(corners, dot_centers, max_distance=10):
    """
    Filter detected corners to only keep those close to known dot centers.
    """
    corners = corners.reshape(-1, 2)  # Reshape to (N, 2)
    filtered_corners = []
    dot_mapping = []
    
    for i, corner in enumerate(corners):
        # Find closest dot center
        distances = np.sqrt(np.sum((dot_centers - corner)**2, axis=1))
        closest_dot_idx = np.argmin(distances)  #returns the index of the closest dot to that specific corner (looped over)
        min_distance = distances[closest_dot_idx]
        
        # Keep corner if it's close enough to a dot
        if min_distance <= max_distance:
            filtered_corners.append(corner)
            dot_mapping.append(closest_dot_idx)
    
    # Convert back to the format expected by Lucas-Kanade
    filtered_corners = np.array(filtered_corners).reshape(-1, 1, 2)
    dot_mapping = np.array(dot_mapping)
    
    return filtered_corners, dot_mapping

def compute_optical_flow(frame1, frame2, p0, is_preprocessed=False):
    """
    Compute optical flow for detected corners.
    """
    if not is_preprocessed:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1  # Already grayscale
        gray2 = frame2
    
    # Adjust parameters based on image type
    if is_preprocessed:
        lk_params = dict(
            winSize=(25, 25),  # Smaller window for preprocessed images
            maxLevel=2,        # Fewer levels needed for binary images
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001),
            minEigThreshold=0.001
        )
    else:
        lk_params = dict(
            winSize=(50, 50),  # Larger window for X-ray images
            maxLevel=4,        # More pyramid levels for handling larger motions
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            minEigThreshold=0.001
        )
    
    p1, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    good_new = p1[status == 1]
    good_old = p0[status == 1]
    
    return good_old, good_new, status

def track_filtered_corners(use_preprocessed=False, use_exact_centers=False):
    """
    Track filtered corners through all frames using optical flow.
    """
    os.makedirs('data/flow_tracking', exist_ok=True)
    
    with open("data/coordinates/grid_centers_1.json", "r") as f:
        dot_data = json.load(f)
    dot_centers = np.array([[dot["x"], dot["y"]] for dot in dot_data])
    
    # Select appropriate frame source
    if use_preprocessed:
        frame_files = sorted(glob.glob('data/frames_prepro/*.png'))
        print("Using preprocessed frames from data/frames_prepro/")
    else:
        frame_files = sorted(glob.glob('data/frames/*.png'))
        print("Using original X-ray frames from data/frames/")
    
    first_frame = cv2.imread(frame_files[0])
    if use_preprocessed:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = first_frame.shape[:2] if use_preprocessed else first_frame.shape[:2]
    
    # Get initial points to track
    if use_exact_centers:
        print("Using exact dot centers from JSON file")
        p0 = dot_centers.astype(np.float32).reshape(-1, 1, 2)
        dot_mapping = np.arange(len(dot_centers))  # Each point maps to itself
    else:
        # Detect and filter corners
        if not use_preprocessed:
            all_corners = detect_corners(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))
        else:
            all_corners = detect_corners(first_frame, is_preprocessed=True)
        p0, dot_mapping = filter_corners_by_dot_centers(all_corners, dot_centers)
    
    tracked_points = []
    n_points = len(p0)
    tracked_points.append(p0.reshape(-1, 2))
    
    for i in range(len(frame_files) - 1):
        if i % 10 == 0:
            print(f"Processing frames {i} → {i+1}")
        
        frame1 = cv2.imread(frame_files[i])
        frame2 = cv2.imread(frame_files[i + 1])
        
        if use_preprocessed:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        good_old, good_new, status = compute_optical_flow(frame1, frame2, p0, is_preprocessed=use_preprocessed)
        
        current_points = np.full((n_points, 2), np.nan)
        current_points[:len(good_new)] = good_new
        
        tracked_points.append(current_points)
        p0 = good_new.reshape(-1, 1, 2)
    
    tracked_points = np.array(tracked_points)
    np.save('data/flow_tracking/filtered_point_trajectories.npy', tracked_points)
    np.save('data/flow_tracking/dot_mapping.npy', dot_mapping)
    
    return tracked_points


def plot_trajectory_directions(trajectories, frame_shape):
    """
    Create a quiver plot showing motion directions.
    """
    plt.figure(figsize=(15, 10))
    
    # Calculate start and end points for each trajectory
    start_points = []
    end_points = []
    
    for point_idx in range(trajectories.shape[1]):
        trajectory = trajectories[:, point_idx, :]
        valid_mask = ~np.isnan(trajectory[:, 0])  #valid_mask is a boolean array of the same length as trajectory, 1 means the point is valid
        valid_trajectory = trajectory[valid_mask]
        
        if len(valid_trajectory) > 1:
            start_points.append(valid_trajectory[0])
            end_points.append(valid_trajectory[-1])
    
    if start_points:
        start_points = np.array(start_points)
        end_points = np.array(end_points)
        
        # Calculate displacement vectors
        dx = end_points[:, 0] - start_points[:, 0]
        dy = end_points[:, 1] - start_points[:, 1]
        
        # Create quiver plot with shorter arrows
        plt.quiver(start_points[:, 0], start_points[:, 1], 
                  dx, dy, angles='xy', scale_units='xy', 
                  scale=20, color='red', alpha=0.5)  # scale=20 for shorter arrows
    
    plt.xlim(0, frame_shape[1])
    plt.ylim(frame_shape[0], 0)  # Invert y-axis for image coordinates
    plt.title('Filtered Corner Motion Direction Vectors')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    return plt.gcf()


def create_trajectory_video(trajectories, frame_shape):
    """
    Create a video showing trajectories building up over time.
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('data/visualizations/trajectory_visualization/filtered_trajectory_buildup.mp4',
                           fourcc, 30.0, (frame_shape[1], frame_shape[0]))
    
    # Create visualization frame
    vis_frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    
    # Create color map for temporal coloring (pink gradient)
    colors = [(1,0,1), (1,0.5,1), (1,0,0.5)]  # Magenta to light pink to dark pink
    for frame_idx in range(1, trajectories.shape[0]):  # Start from frame 1
        frame_copy = vis_frame.copy()
        
        # Draw only frame-to-frame motion vectors (not cumulative trajectories)
        for point_idx in range(trajectories.shape[1]):
            # Get current and previous frame positions
            current_pos = trajectories[frame_idx, point_idx, :]
            prev_pos = trajectories[frame_idx-1, point_idx, :]
            
            # Only draw if both positions are valid (not NaN)
            if not (np.isnan(current_pos[0]) or np.isnan(prev_pos[0])):
                # Draw motion vector from previous to current position
                pt1 = tuple(map(int, prev_pos))
                pt2 = tuple(map(int, current_pos))
                cv2.line(frame_copy, pt1, pt2, (0, 255, 255), 1)  # Yellow motion vectors
                cv2.circle(frame_copy, pt2, 2, (0, 255, 0), -1)  # Green current position
        
        # Add frame number and corner count
        cv2.putText(frame_copy, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_copy, f"Motion Vectors: {trajectories.shape[1]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        video.write(frame_copy)
    
    video.release()

    return None

def visualize_trajectories(trajectories, use_preprocessed=False):
    """
    Create all trajectory visualizations.
    """
    print("\n=== STEP 2: Creating Trajectory Visualizations ===")
    
    # Read first frame to get dimensions
    if use_preprocessed:
        frame_files = sorted(glob.glob('data/frames_prepro/*.png'))
        first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    else:
        first_frame = cv2.imread('data/frames/frame_1.png', cv2.IMREAD_GRAYSCALE)
    frame_shape = first_frame.shape
    
    # Create output directory
    os.makedirs('data/visualizations/trajectory_visualization', exist_ok=True)
    
    print(f"Processing {trajectories.shape[1]} filtered corners")
    
    # Create direction plot
    print("Creating direction plot...")
    direction_fig = plot_trajectory_directions(trajectories, frame_shape)
    direction_fig.savefig('data/visualizations/trajectory_visualization/filtered_motion_directions.png')
    plt.close()
    
    # Create trajectory buildup video
    print("Creating filtered trajectory buildup video...")
    create_trajectory_video(trajectories, frame_shape)
    
    print("\nTrajectory visualization complete! Results saved in data/visualizations/trajectory_visualization/")
    print("- Filtered motion directions: filtered_motion_directions.png")
    print("- Filtered trajectory buildup video: filtered_trajectory_buildup.mp4")

def plot_consecutive_frames(trajectories, frame_start_idx=10, use_preprocessed=False):
    """
    Create a professional plot of three consecutive frames with optical flow vectors.
    Suitable for thesis visualization.
    """
    # Read three consecutive frames
    if use_preprocessed:
        frame_files = sorted(glob.glob('data/frames_prepro/*.png'))
    else:
        frame_files = sorted(glob.glob('data/frames/*.png'))
    
    frames = []
    for i in range(frame_start_idx, frame_start_idx + 3):
        frame = cv2.imread(frame_files[i])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        print(f"Frame {i} shape: {frame.shape}")  # Debug print
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set equal aspect ratio for all subplots
    for ax in axes:
        ax.set_aspect('equal')
    
    # Plot each frame with its flow vectors
    for idx, (ax, frame) in enumerate(zip(axes, frames)):
        # Show the frame
        ax.imshow(frame, cmap='gray', aspect='equal')
        
        # Get points and vectors for this frame
        current_points = trajectories[frame_start_idx + idx]
        next_points = trajectories[frame_start_idx + idx + 1]
        
        # Plot flow vectors
        for curr, next_p in zip(current_points, next_points):
            if not (np.isnan(curr[0]) or np.isnan(next_p[0])):
                # Draw arrow
                dx = next_p[0] - curr[0]
                dy = next_p[1] - curr[1]
                ax.arrow(curr[0], curr[1], dx, dy,
                        color='yellow', alpha=0.6,
                        head_width=3, head_length=5,
                        length_includes_head=True)
                # Draw point
                ax.plot(curr[0], curr[1], 'go', markersize=2)
        
        # Add frame number inside the image
        ax.text(10, 25, f'Frame {frame_start_idx + idx}', 
                color='white', fontsize=10, 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3))
        
        ax.axis('off')
    
    # Ensure all subplots have exactly the same size
    plt.subplots_adjust(wspace=0.01, left=0, right=1, bottom=0, top=1)
    
    # Save with consistent dimensions
    os.makedirs('data/visualizations/thesis_figures', exist_ok=True)
    base_filename = f'consecutive_frames_flow_{frame_start_idx}_{frame_start_idx+2}'
    plt.savefig(f'data/visualizations/thesis_figures/{base_filename}.pdf', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'data/visualizations/thesis_figures/{base_filename}.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
