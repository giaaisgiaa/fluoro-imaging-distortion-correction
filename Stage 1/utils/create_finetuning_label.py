import os
import json
import numpy as np
from PIL import Image
import random
import shutil
from tqdm import tqdm

def ensure_directories():
    """Create and clean necessary directories"""
    # Base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(base_dir, "data", "frames")
    coords_dir = os.path.join(base_dir, "data", "coordinates_finetuning")
    
    # Create finetuning labels directory
    finetune_dir = os.path.join(base_dir, "data", "finetuning_labels")
    
    # Remove entire finetuning_labels directory if it exists
    if os.path.exists(finetune_dir):
        print(f"Cleaning up existing directory: {finetune_dir}")
        shutil.rmtree(finetune_dir)
    
    # Create fresh directories
    os.makedirs(finetune_dir)
    frames_out_dir = os.path.join(finetune_dir, "frames")
    heatmaps_dir = os.path.join(finetune_dir, "heatmaps")
    os.makedirs(frames_out_dir)
    os.makedirs(heatmaps_dir)
    
    print(f"Created fresh directories:")
    print(f"- {frames_out_dir}")
    print(f"- {heatmaps_dir}")
    
    return frames_dir, coords_dir, frames_out_dir, heatmaps_dir

def create_gaussian_heatmap(centers, image_size, dot_radius):
    """Create a heatmap with Gaussian blobs at dot centers using same method as DotDataset"""
    # Create target heatmap (Gaussian blobs at dot centers)
    heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    for dot in centers:
        x, y = dot['x'], dot['y']
        # Create a small Gaussian blob at each dot center
        for i in range(max(0, x - dot_radius * 3), min(image_size[0], x + dot_radius * 3)):
            for j in range(max(0, y - dot_radius * 3), min(image_size[1], y + dot_radius * 3)):
                dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                # Gaussian with sigma = dot_radius/2
                heatmap[j, i] = max(heatmap[j, i], 
                               np.exp(-dist**2 / (2 * (dot_radius/2)**2)))
    return heatmap

def process_frames(frames_dir, coords_dir, frames_out_dir, heatmaps_dir, num_frames=10, dot_radius=10):
    """Process random frames and create corresponding heatmaps"""
    # Get all frame files
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    
    # Select random frames
    selected_frames = random.sample(frame_files, min(num_frames, len(frame_files)))
    print(f"\nSelected {len(selected_frames)} frames for finetuning:")
    for i, frame in enumerate(selected_frames, 1):
        print(f"{i}. {frame}")
    
    # Process each selected frame
    for frame_file in tqdm(selected_frames, desc="Processing frames"):
        # Extract frame number
        frame_num = frame_file.replace('frame_', '').replace('.png', '')
        
        # Load frame
        frame_path = os.path.join(frames_dir, frame_file)
        frame = Image.open(frame_path).convert('L')  # Convert to grayscale
        image_size = frame.size  # (width, height)
        
        # Load corresponding coordinates
        coord_file = f'grid_centers_{frame_num}.json'
        coord_path = os.path.join(coords_dir, coord_file)
        
        try:
            with open(coord_path, 'r') as f:
                centers = json.load(f)  # Centers are directly a list of coordinates
            
            # Create heatmap using same method as DotDataset
            heatmap = create_gaussian_heatmap(centers, image_size, dot_radius)
            
            # Save frame (normalized like in DotDataset)
            frame_array = np.array(frame, dtype=np.float32) / 255.0  # Normalize to 0-1
            frame_norm = Image.fromarray((frame_array * 255).astype(np.uint8))
            frame_norm.save(os.path.join(frames_out_dir, frame_file))
            
            # Save heatmap as PNG
            heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_img.save(os.path.join(heatmaps_dir, frame_file))
            
            # Save heatmap data as NPY for precise values
            np.save(os.path.join(heatmaps_dir, frame_file.replace('.png', '.npy')), heatmap)
            
            # Save metadata
            metadata = {
                'image_size': list(image_size),
                'dot_radius': dot_radius,
                'num_dots': len(centers),
                'dots': centers
            }
            with open(os.path.join(heatmaps_dir, frame_file.replace('.png', '_meta.json')), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Processed {frame_file}: Found {len(centers)} dots")
            
        except FileNotFoundError:
            print(f"Warning: Coordinates file not found for {frame_file} (looking for {coord_file})")
            continue
        except Exception as e:
            print(f"Error processing {frame_file}: {e}")
            continue

def main():
    # Create directories
    frames_dir, coords_dir, frames_out_dir, heatmaps_dir = ensure_directories()
    
    # Process frames
    process_frames(frames_dir, coords_dir, frames_out_dir, heatmaps_dir)
    
    print("\nProcessing complete!")
    print(f"Finetuning data saved in:")
    print(f"- Frames: {frames_out_dir}")
    print(f"- Heatmaps: {heatmaps_dir}")

if __name__ == "__main__":
    main() 