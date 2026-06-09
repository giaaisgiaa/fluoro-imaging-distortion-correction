import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from model.dot_cnn import UNet
from scipy import ndimage
from training.train_model import get_device
from datetime import datetime


def ensure_directories(script_dir, run_dir=None):
    """
    Create necessary directories for outputs
    If run_dir is not provided, creates a new timestamped run directory
    """
    if run_dir is None:
        # Create a new run directory with timestamp
        timestamp = datetime.now().strftime("%d%m%y_%H%M")
        runs_dir = os.path.join(script_dir, "../runs")
        os.makedirs(runs_dir, exist_ok=True)
        run_dir = os.path.join(runs_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
    else:
        # Use the provided run directory
        os.makedirs(run_dir, exist_ok=True)
    
    # Create calibration grid predictions directory
    grid_pred_dir = os.path.join(run_dir, "calibration_grid_predictions")
    os.makedirs(grid_pred_dir, exist_ok=True)
    
    # Create subdirectories for different visualizations
    grid_centers_dir = os.path.join(grid_pred_dir, "white_dots")
    grid_centers_red_dir = os.path.join(grid_pred_dir, "red_dots")
    coordinates_dir = os.path.join(grid_pred_dir, "coordinates")
    
    # Create and clean directories
    for dir_path in [grid_centers_dir, grid_centers_red_dir, coordinates_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            # Clean up existing files
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print(f"Cleaned up directory: {dir_path}")
    
    return grid_centers_dir, grid_centers_red_dir, coordinates_dir, run_dir

def load_model(model_path, device=None):
    """
    Load the model once and return it
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS device: {device}")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {device}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU device: {device}")

    # Load model
    model = UNet()
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # If saved with model_state_dict and metrics
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If saved as just the state dict
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    model.to(device)
    model.eval()
    
    return model, device



# Function to predict dot locations from heatmap
def find_dot_centers(heatmap, threshold=0.5):
    # Convert heatmap to numpy array and threshold
    heatmap_np = heatmap.squeeze().cpu().numpy()
    binary = (heatmap_np > threshold).astype(np.uint8)
    
    # Find connected components (blobs)
    labeled, num_features = ndimage.label(binary)
    
    # Find center of mass for each blob --> center of the dot
    centers = []
    for i in range(1, num_features + 1):
        y, x = ndimage.center_of_mass(heatmap_np, labeled, i)
        centers.append({"x": int(x), "y": int(y)})
    
    return centers




def infer_on_image(image_path, original_image_path, model, device, output_path_red, output_path_white, threshold=0.1):
    """
    Run inference on an image and create two visualizations:
    1. Red dots on original image (saved to grid_centers_red)
    2. White dots on black background (saved to grid_centers)
    """
    # Load image and convert to grayscale
    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')
    
    # Apply same transforms as in training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Store original image for visualization
    orig_image = np.array(image)
    
    # Process the image at original size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        pred_heatmap = model(image_tensor)
    
    # Detect dots
    pred_centers = find_dot_centers(pred_heatmap[0], threshold=threshold)
    
    # Visualization 1: Red dots on original image
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_image, cmap='gray', aspect='equal')
    
    # Plot dots
    for center in pred_centers:
        plt.plot(center['x'], center['y'], 'ro', markersize=2, alpha=0.3)
    
    plt.title(f"Detected {len(pred_centers)} dots (threshold={threshold})")
    plt.axis('off')
    
    # Save visualization with red dots
    plt.tight_layout()
    plt.savefig(output_path_red)
    plt.close()
    
    # Visualization 2: White dots on black background
    plt.figure(figsize=(10, 10))
    black_bg = np.zeros_like(orig_image)
    plt.imshow(black_bg, cmap='gray', aspect='equal')
    
    # Plot dots
    for center in pred_centers:
        plt.plot(center['x'], center['y'], 'wo', markersize=2, alpha=1.0)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path_white, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return pred_centers




def process_all_frames(frames_dir, model_path, threshold=0.1, num_frames=None, run_dir=None):
    """
    Process frames in the given directory using the trained model
    Args:
        frames_dir: Directory containing the frames
        model_path: Path to the trained model
        threshold: Threshold for dot detection
        num_frames: Number of random frames to process (None for all frames)
        run_dir: Optional specific run directory to use (if None, creates new one)
    """
    # Get device and load model
    device = get_device()
    model, device = load_model(model_path, device)
    if model is None:
        return
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure output directories exist and are clean
    grid_centers_dir, grid_centers_red_dir, coordinates_dir, run_dir = ensure_directories(script_dir, run_dir)
    
    # Get all frame files and sort them numerically
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    frame_files.sort(key=lambda x: int(x.replace('frame_', '').replace('.png', '')))
    
    # Select random frames if num_frames is specified
    if num_frames is not None and num_frames < len(frame_files):
        import random
        frame_files = random.sample(frame_files, num_frames)
        print(f"\nRandomly selected {num_frames} frames to process:")
        for i, f in enumerate(frame_files, 1):
            print(f"{i}. {f}")
    else:
        print(f"\nProcessing all {len(frame_files)} frames...")
    
    for frame_file in frame_files:
        # Get frame number
        frame_num = int(frame_file.replace('frame_', '').replace('.png', ''))
        
        # Construct paths
        frame_path = os.path.join(frames_dir, frame_file)
        output_path_red = os.path.join(grid_centers_red_dir, frame_file)
        output_path_white = os.path.join(grid_centers_dir, frame_file)
        json_path = os.path.join(coordinates_dir, f'grid_centers_{frame_num}.json')
        
        print(f"\nProcessing {frame_file}...")
        
        # Run inference and get dot centers
        pred_centers = infer_on_image(frame_path, frame_path, model, device, 
                                    output_path_red, output_path_white, threshold)
        
        # Save coordinates to JSON - directly as a list, not wrapped in {'dots': ...}
        with open(json_path, 'w') as f:
            json.dump(pred_centers, f, indent=2)
        
        print(f"Found {len(pred_centers)} dots")
    
    print("\nProcessing complete!")
    print(f"Results saved in: {grid_centers_dir}")
    print(f"Run directory: {run_dir}")

