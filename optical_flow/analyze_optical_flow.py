import numpy as np
import cv2
from utils.error_metrics import OpticalFlowMetrics, print_metrics_table
from utils.detect_and_filter_corners import detect_corners, filter_corners_by_dot_centers
from utils.visualize_filtered_trajectories import (
    track_filtered_corners,
    visualize_trajectories,
    plot_consecutive_frames,
    compute_optical_flow
)
import matplotlib.pyplot as plt
import os
import json
import glob

def ensure_directories_exist():
    """
    Create all required directories if they don't exist.
    """
    required_dirs = [
        'data/coordinates',
        'data/flow_tracking',
        'data/visualizations/optical_flow_analysis',
        'data/visualizations/trajectory_visualization',
        'data/visualizations/thesis_figures',
        'frames',
        'frames_prepro'
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def load_trajectories():
    """
    Load the optical flow trajectories and true dot positions.
    """
    # Load true trajectories from JSON files
    coord_files = sorted(glob.glob('data/coordinates/grid_centers_*.json'))
    
    # Load all coordinates first to determine the maximum number of points
    all_points = []
    print(f"Loading {len(coord_files)} frames of true coordinates...")
    
    for file_path in coord_files:
        with open(file_path, 'r') as f:
            frame_data = json.load(f)
            points = np.array([[point['x'], point['y']] for point in frame_data])
            all_points.append(points)
    
    # Find the maximum number of points across all frames
    max_points = max(points.shape[0] for points in all_points)
    num_frames = len(coord_files)
    
    # Initialize arrays with NaN values
    true_trajectories = np.full((num_frames, max_points, 2), np.nan)
    
    # Fill the array with available points
    for i, points in enumerate(all_points):
        true_trajectories[i, :points.shape[0]] = points
    
    # Load optical flow predictions and dot mapping
    if os.path.exists('data/flow_tracking/filtered_point_trajectories.npy') and os.path.exists('data/flow_tracking/dot_mapping.npy'):
        predicted_trajectories = np.load('data/flow_tracking/filtered_point_trajectories.npy')
        dot_mapping = np.load('data/flow_tracking/dot_mapping.npy')
        print(f"Loaded optical flow predictions, shape: {predicted_trajectories.shape}")
        print(f"Loaded dot mapping, shape: {dot_mapping.shape}")
        
        # Rearrange true trajectories according to dot mapping
        mapped_true_trajectories = np.full_like(predicted_trajectories, np.nan)
        for i, mapping_idx in enumerate(dot_mapping):
            if mapping_idx < true_trajectories.shape[1]:
                mapped_true_trajectories[:, i] = true_trajectories[:, mapping_idx]
        true_trajectories = mapped_true_trajectories
        
    else:
        print("Warning: No optical flow predictions or dot mapping found. Using dummy data...")
        predicted_trajectories = true_trajectories + np.random.normal(0, 5, true_trajectories.shape)
    
    print(f"Data loaded successfully:")
    print(f"True trajectories shape: {true_trajectories.shape}")
    print(f"Predicted trajectories shape: {predicted_trajectories.shape}")
    print(f"Number of frames: {num_frames}")
    print(f"Maximum points per frame: {max_points}")
    
    return predicted_trajectories, true_trajectories

def plot_error_over_time(predicted_trajectories, true_trajectories):
    """
    Create plots showing how errors evolve over time.
    """
    metrics_calculator = OpticalFlowMetrics(threshold_pixels=5.0)
    
    # Calculate frame-by-frame metrics
    frame_errors = []
    frame_loss_rates = []
    
    for t in range(len(predicted_trajectories)):
        # Get valid points
        pred_valid = predicted_trajectories[t]
        true_valid = true_trajectories[t]
        valid_mask = ~np.isnan(pred_valid).any(axis=1) & ~np.isnan(true_valid).any(axis=1)
        
        if np.sum(valid_mask) > 0:
            pred_valid = pred_valid[valid_mask]
            true_valid = true_valid[valid_mask]
            
            mee, _ = metrics_calculator.mean_endpoint_error(pred_valid, true_valid)
            tlr, _ = metrics_calculator.tracking_loss_rate(pred_valid, true_valid)
            
            frame_errors.append(mee)
            frame_loss_rates.append(tlr)
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(frame_errors, 'b-', label='Error')
    plt.axhline(y=np.mean(frame_errors), color='r', linestyle='--', label=f'Mean: {np.mean(frame_errors):.2f}px')
    plt.title('Mean Endpoint Error Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Error (pixels)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(frame_loss_rates, 'g-', label='Loss Rate')
    plt.axhline(y=np.mean(frame_loss_rates), color='r', linestyle='--', label=f'Mean: {np.mean(frame_loss_rates):.1f}%')
    plt.title('Tracking Loss Rate Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Loss Rate (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('data/visualizations/optical_flow_analysis', exist_ok=True)
    plt.savefig('data/visualizations/optical_flow_analysis/error_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Ensure all required directories exist
    ensure_directories_exist()
    
    # First: Corner Detection and Filtering
    print("\n=== STEP 1: Corner Detection and Filtering ===")
    
    # Ask user for tracking method preference
    while True:
        print("\nChoose tracking method:")
        print("1: Original X-ray images with Shi-Tomasi corner detection")
        print("2: Preprocessed binary images with Shi-Tomasi corner detection")
        print("3: Use exacted dot centers (from JSON) as features")
        choice = input("Enter choice (1, 2, or 3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    use_preprocessed = choice in ['2', '3']
    use_exact_centers = (choice == '3')
    
    # Step 1: Track filtered corners
    trajectories = track_filtered_corners(use_preprocessed, use_exact_centers)
    
    # Step 2: Visualize trajectories
    visualize_trajectories(trajectories, use_preprocessed)
    
    # Step 3: Create thesis figures for different frame sequences
    if use_preprocessed:
        frame_starts = [1, 10, 19, 28]  # Adjusted for preprocessed frame indices
    else:
        frame_starts = [10, 30, 60, 90]  # Original frame indices
    
    for start_idx in frame_starts:
        plot_consecutive_frames(trajectories, frame_start_idx=start_idx, use_preprocessed=use_preprocessed)
        print(f"Created visualization for frames {start_idx}-{start_idx+2}")
    
    # Step 4: Calculate and display metrics
    predicted_trajectories, true_trajectories = load_trajectories()
    
    if predicted_trajectories is not None and true_trajectories is not None:
        metrics_calculator = OpticalFlowMetrics(threshold_pixels=5.0)
        metrics = metrics_calculator.calculate_all_metrics(predicted_trajectories, true_trajectories)
        
        # Print results
        print_metrics_table(metrics)
        
        # Create visualizations
        plot_error_over_time(predicted_trajectories, true_trajectories)
    
    print("\n=== Pipeline Complete! ===")
    print("All results saved in:")
    print("- data/flow_tracking/ (trajectory data)")
    print("- data/visualizations/trajectory_visualization/ (visualizations)")
    print("- data/visualizations/thesis_figures/ (publication-quality figures)")
    print("- data/visualizations/optical_flow_analysis/ (error analysis)")

if __name__ == "__main__":
    main() 