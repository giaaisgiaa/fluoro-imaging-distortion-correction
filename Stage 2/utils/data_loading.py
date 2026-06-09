import numpy as np
import json
import os
from tqdm import tqdm

_pbar = None

def init_loading_progress():
    global _pbar
    _pbar = tqdm(total=150, desc="Loading frames")

def load_grid_centers(frame_number, coordinates_dir="data/coordinates"):
    """
    Load grid centers from JSON file and convert to numpy array.
    If NPY version exists in data/coordinates_numpy/, load that instead.
    """
    global _pbar
    if _pbar is None:
        init_loading_progress()
        
    # Check if NPY version exists
    npy_dir = "data/coordinates_numpy"
    os.makedirs(npy_dir, exist_ok=True)
    npy_path = os.path.join(npy_dir, f"grid_centers_{frame_number}.npy")
    
    if os.path.exists(npy_path):
        _pbar.update(1)
        return np.load(npy_path)
        
    # Load from JSON and convert
    filename = f"grid_centers_{frame_number}.json"
    filepath = os.path.join(coordinates_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Coordinate file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert JSON data to numpy array
    points = np.array([[point['x'], point['y']] for point in data])
    
    # Save as NPY for future use
    np.save(npy_path, points)
    
    # Update progress bar
    _pbar.update(1)
    
    return points
