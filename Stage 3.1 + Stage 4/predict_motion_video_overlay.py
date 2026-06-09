import os
import numpy as np
import cv2
from utils.dataset_read import load_frame_datasets
from non_stationary_models.forward_rbf_ensemble import ForwardRBFEnsemble
import glob

def _to_pixels(P, W, H):
    """Convert coordinates to pixel positions.
    
    Parameters:
    -----------
    P : array (N, 2)
        Points to convert
    W, H : int
        Width and height of the output frame in pixels
    """
    # Direct mapping assuming coordinates are already in pixel space
    px = P[:, 0].astype(np.int32)
    py = P[:, 1].astype(np.int32)
    
    # Ensure coordinates are within frame bounds
    px = np.clip(px, 0, W-1)
    py = np.clip(py, 0, H-1)
    
    return px, py

def create_overlay_prediction_video(n_random_points=2000, fps=30):
    """
    Create a video showing RBF predictions overlaid on real grid images.
    Uses Ensemble RBF with sequential mode for accurate step-by-step prediction.
    """
    print("="*60)
    print(f"CREATING OVERLAY PREDICTION VIDEO WITH {n_random_points} RANDOM POINTS")
    print("Using Ensemble RBF in sequential mode")
    print("="*60)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_frame_datasets("data/dot_trajectories.json")
    
    # Convert to tensor format
    print("Converting data to tensor format...")
    frames = sorted(datasets.keys(), key=lambda k: int(k.split()[-1]))
    
    # Get first frame to determine number of dots
    first_frame = frames[0]
    n_dots = datasets[first_frame].shape[0]
    print(f"\nNumber of dots per frame: {n_dots}")
    
    # Initialize array for all frames
    n_frames = len(frames)
    data = np.zeros((n_frames, n_dots, 2), dtype=np.float32)
    
    # Fill data frame by frame
    for i, frame in enumerate(frames):
        data[i] = datasets[frame]
    
    print(f"\nFinal data shape: {data.shape}, dtype: {data.dtype}")
    
    # Set random seed for consistent ensemble composition
    np.random.seed(42)
    
    # Train the Ensemble RBF model in sequential mode
    print("Training Ensemble RBF model in sequential mode...")
    model = ForwardRBFEnsemble(
        n_models=5,
        kernel="thin_plate_spline",
        neighbors=80,
        smoothing=1e-3,
        time_weight=1.0,
        standardize=True
    ).fit_tensor(data, t0=0.0, dt=1.0)  # Train the ensemble
    
    # Get initial dot coordinates (first frame)
    real_dots = data[0].astype(np.float32)
    print(f"Using {len(real_dots)} real dots from first frame")
    
    # Create random points using convex hull sampling
    print(f"Generating {n_random_points} random points...")
    def sample_random_convex_points(X_base, n_points=5000, k=4, seed=None):
        rng = np.random.default_rng(seed)
        idx = rng.integers(len(X_base), size=(n_points, k))
        W = rng.dirichlet(np.ones(k), size=n_points)
        return (X_base[idx] * W[..., None]).sum(axis=1)
    
    random_points = sample_random_convex_points(real_dots, n_random_points, k=4, seed=42).astype(np.float32)
    
    # Pre-calculate all predictions for efficiency
    print("\nPre-calculating all predictions...")
    n_frames = len(frames)
    grid_predictions = []
    random_predictions = []
                       
    # Store initial positions
    grid_predictions.append(data[0].copy())  # First frame real positions
    random_predictions.append(random_points.copy())
    
    # Calculate predictions for all frames using sequential mode
    for t in range(n_frames - 1):
        # Get current frame's actual positions for grid points
        current_grid = data[t].copy()
        # For random points, use previous prediction
        current_random = random_predictions[-1].copy()
        
        # Predict one step forward
        next_grid = model.forward_step(current_grid, float(t))
        next_random = model.forward_step(current_random, float(t))
        
        grid_predictions.append(next_grid)
        random_predictions.append(next_random)
    
    # Convert predictions to numpy arrays for faster access
    grid_predictions = np.array(grid_predictions)
    random_predictions = np.array(random_predictions)
    print("Predictions pre-calculated!")
    
    # Load real grid images
    print("\nLoading real grid images...")
    frame_files = sorted(glob.glob("frames/*.png"), 
                        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    if not frame_files:
        print("No frame images found in frames/ directory!")
        return
    
    # Print first few frame filenames to check numbering
    print("\nFirst 5 frame files:")
    for f in frame_files[:5]:
        print(f"  {os.path.basename(f)}")
    
    print("\nFirst 5 prediction frames:")
    for f in frames[:5]:
        print(f"  {f}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_frame.shape
    print(f"\nFrame dimensions from image: {width}x{height}")
    
    # No need for bounds calculation since we're working in pixel coordinates
    
    # Setup video writer
    output_path = "overlay_prediction_with_random_dots.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    print("\nGenerating video frames...")
    n_frames = min(len(frame_files), len(frames))
    
    # Create frame number mapping if needed
    frame_numbers = [int(''.join(filter(str.isdigit, os.path.basename(f)))) for f in frame_files]
    print(f"\nFrame number ranges:")
    print(f"Original frames: {min(frame_numbers)} to {max(frame_numbers)}")
    print(f"Prediction frames: 1 to {len(frames)}")
    
    # Adjust frame indices if needed
    frame_step = max(frame_numbers) // len(frames)
    if frame_step > 1:
        print(f"\nAdjusting for frame rate difference (step: {frame_step})")
    
    for i in range(n_frames):
        frame_idx = i
        img_idx = i * frame_step if frame_step > 1 else i
        
        if img_idx >= len(frame_files):
            break
            
        print(f"Processing frame {frame_idx + 1}/{n_frames} (image {img_idx + 1})...")
        
        # Read and convert frame to color (for overlay)
        frame_image = cv2.imread(frame_files[img_idx], cv2.IMREAD_GRAYSCALE)
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_GRAY2BGR)
        
        # Get pre-calculated predictions for this frame
        predicted_grid_positions = grid_predictions[frame_idx]
        predicted_random_positions = random_predictions[frame_idx]
        
        # Draw predicted grid positions in magenta (high contrast on grayscale)
        px_pred_grid, py_pred_grid = _to_pixels(predicted_grid_positions, width, height)
        
        # Print coordinate ranges for first frame only
        if frame_idx == 0:
            print("\nFirst frame coordinate analysis:")
            print(f"Original frame size: {width}x{height}")
            print(f"First 5 predicted grid points (raw coordinates):")
            for i in range(5):
                print(f"Point {i}: ({predicted_grid_positions[i][0]:.1f}, {predicted_grid_positions[i][1]:.1f})")
            print(f"First 5 predicted grid points (pixel coordinates):")
            for i in range(5):
                print(f"Point {i}: ({px_pred_grid[i]}, {py_pred_grid[i]})")
        
        # Draw original grid points in green for comparison
        for x, y in zip(px_pred_grid, py_pred_grid):
            if 0 <= x < width and 0 <= y < height:
                # Draw larger circles in white first for visibility
                cv2.circle(frame_image, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
                # Draw predicted points in magenta
                cv2.circle(frame_image, (x, y), 3, (255, 0, 255), -1, cv2.LINE_AA)  # Magenta
        
        # Draw predicted random positions in bright green (high contrast on grayscale)
        px_pred_random, py_pred_random = _to_pixels(predicted_random_positions, width, height)
        for x, y in zip(px_pred_random, py_pred_random):
            if 0 <= x < width and 0 <= y < height:
                # Draw larger circles in white first for visibility
                cv2.circle(frame_image, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
                # Draw random points in bright green
                cv2.circle(frame_image, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)  # Bright green (BGR format)

        # Add text overlay
        cv2.putText(frame_image, f"Frame {frame_idx + 1}/{n_frames} - Ensemble RBF", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_image, f"Magenta: {len(predicted_grid_positions)} grid predictions", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_image, f"Green: {len(predicted_random_positions)} random predictions", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write frame
        video_writer.write(frame_image)
    
    # Release video writer
    video_writer.release()
    
    print(f"\n{'='*60}")
    print("OVERLAY PREDICTION VIDEO CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Saved: {output_path}")
    print(f"Video dimensions: {width} x {height}")
    print(f"FPS: {fps}, Duration: {n_frames/fps:.1f} seconds")
    print(f"Total frames: {n_frames}")
    print(f"Grid points: {len(real_dots)}")
    print(f"Random points: {len(random_points)}")
    
    return output_path

if __name__ == "__main__":
    print("="*60)
    print("OVERLAY PREDICTION VIDEO GENERATOR")
    print("="*60)
    
    # Ask user for number of random points
    random_points_input = input("Enter number of random points to predict (default: 2000): ").strip()
    n_random_points = int(random_points_input) if random_points_input else 2000
    
    # Ask for FPS
    fps_input = input("Enter desired FPS (default: 30): ").strip()
    fps = int(fps_input) if fps_input else 30
    
    print(f"\nCreating overlay video with {n_random_points} random points at {fps} FPS...")
    create_overlay_prediction_video(n_random_points, fps) 