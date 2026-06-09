import numpy as np
from scipy.spatial.distance import cdist
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

def find_dot_centers(heatmap, threshold=0.5):
    """Find dot centers from a heatmap using local maxima detection with non-maximum suppression."""
    # Use a larger window size for maximum filtering to avoid multiple detections per dot
    window_size = 11  # Increased from 3 to 11 to cover typical dot size
    
    # Find local maxima
    max_filtered = ndimage.maximum_filter(heatmap, size=window_size)
    maxima = (heatmap == max_filtered) & (heatmap > threshold)
    
    # Get coordinates and convert to (x,y) format
    coords = np.column_stack(np.where(maxima))
    centers = np.flip(coords, axis=1)  # Convert from (row,col) to (x,y)
    
    # Convert to list of dictionaries format
    centers_list = [{"x": int(x), "y": int(y)} for x, y in centers]
    
    return centers_list

def calculate_detection_metrics(pred_heatmap, true_centers, threshold=0.1, max_distance=10):
    """Calculate detection accuracy metrics."""
    try:
        # Find predicted centers
        pred_centers = find_dot_centers(pred_heatmap, threshold)
        
        # Convert to numpy arrays
        true_centers = np.array(true_centers)
        pred_centers = np.array([[center['x'], center['y']] for center in pred_centers])
        
        # Handle edge cases
        num_true = len(true_centers)
        num_detected = len(pred_centers)
        
        if num_detected == 0 or num_true == 0:
            return {
                'mean_distance': float('inf') if num_true > 0 else 0.0,
                'precision': 0.0 if num_detected > 0 else 1.0,
                'recall': 0.0 if num_true > 0 else 1.0,
                'num_true': num_true,
                'num_detected': num_detected,
                'num_correct': 0
            }
        
        # Calculate distances and find optimal matching
        distances = cdist(true_centers, pred_centers)
        true_idx, pred_idx = linear_sum_assignment(distances)
        
        # Count correct detections
        matched_distances = distances[true_idx, pred_idx]
        correct_matches = matched_distances <= max_distance
        num_correct = int(np.sum(correct_matches))
        
        # Calculate metrics (with safety checks)
        precision = float(num_correct) / float(num_detected) if num_detected > 0 else 0.0
        recall = float(num_correct) / float(num_true) if num_true > 0 else 0.0
        mean_distance = float(np.mean(matched_distances[correct_matches])) if num_correct > 0 else float('inf')
        
        return {
            'mean_distance': mean_distance,
            'precision': precision,
            'recall': recall,
            'num_true': num_true,
            'num_detected': num_detected,
            'num_correct': num_correct
        }
    except Exception as e:
        print(f"Warning: Error calculating metrics: {str(e)}")
        return {
            'mean_distance': float('inf'),
            'precision': 0.0,
            'recall': 0.0,
            'num_true': 0,
            'num_detected': 0,
            'num_correct': 0
        }
