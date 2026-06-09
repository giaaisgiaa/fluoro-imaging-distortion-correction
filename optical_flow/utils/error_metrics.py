import numpy as np
from typing import Tuple, List

class OpticalFlowMetrics:
    def __init__(self, threshold_pixels: float = 5.0):
        """
        Initialize metrics calculator.
        
        Args:
            threshold_pixels: Threshold in pixels to consider a track as "lost"
        """
        self.threshold_pixels = threshold_pixels

    def mean_endpoint_error(self, 
                          predicted_positions: np.ndarray, 
                          true_positions: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Mean Endpoint Error (MEE) and its standard deviation.
        
        Args:
            predicted_positions: Array of predicted positions (N, 2) where N is number of points
            true_positions: Array of true positions (N, 2)
            
        Returns:
            Tuple of (mean_error, std_error) in pixels
        """
        # Calculate Euclidean distance for each point
        errors = np.sqrt(np.sum((predicted_positions - true_positions)**2, axis=1))
        
        return np.mean(errors), np.std(errors)

    def tracking_loss_rate(self, 
                          predicted_positions: np.ndarray, 
                          true_positions: np.ndarray) -> Tuple[float, float]:
        """
        Calculate percentage of lost tracks (points with error > threshold).
        
        Args:
            predicted_positions: Array of predicted positions (N, 2)
            true_positions: Array of true positions (N, 2)
            
        Returns:
            Tuple of (loss_rate_percentage, std_deviation)
        """
        # Calculate errors
        errors = np.sqrt(np.sum((predicted_positions - true_positions)**2, axis=1))
        
        # Calculate loss rate
        lost_tracks = np.sum(errors > self.threshold_pixels)
        loss_rate = (lost_tracks / len(errors)) * 100
        
        # Calculate standard deviation using binomial distribution
        std_dev = np.sqrt((loss_rate * (100 - loss_rate)) / len(errors))
        
        return loss_rate, std_dev

    def maximum_endpoint_error(self, 
                             predicted_positions: np.ndarray, 
                             true_positions: np.ndarray) -> float:
        """
        Calculate maximum endpoint error.
        
        Args:
            predicted_positions: Array of predicted positions (N, 2)
            true_positions: Array of true positions (N, 2)
            
        Returns:
            Maximum error in pixels
        """
        errors = np.sqrt(np.sum((predicted_positions - true_positions)**2, axis=1))
        return np.max(errors)

    def temporal_consistency(self, 
                           trajectories: np.ndarray) -> Tuple[float, float]:
        """
        Calculate temporal consistency of trajectories.
        
        Args:
            trajectories: Array of shape (T, N, 2) where:
                        T is number of frames
                        N is number of points
                        2 is (x,y) coordinates
            
        Returns:
            Tuple of (mean_consistency, std_consistency)
        """
        # Calculate velocities (differences between consecutive positions)
        velocities = trajectories[1:] - trajectories[:-1]
        
        # Calculate acceleration (differences in velocities)
        accelerations = velocities[1:] - velocities[:-1]
        
        # Calculate consistency measure (mean magnitude of acceleration)
        consistency = np.sqrt(np.sum(accelerations**2, axis=2))
        
        return np.mean(consistency), np.std(consistency)

    def calculate_all_metrics(self, 
                            predicted_trajectories: np.ndarray, 
                            true_trajectories: np.ndarray) -> dict:
        """
        Calculate all metrics for the entire sequence.
        
        Args:
            predicted_trajectories: Array of shape (T, N, 2) for predicted positions
            true_trajectories: Array of shape (T, N, 2) for true positions
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Calculate metrics for each frame (excluding first and last)
        mee_values = []
        tlr_values = []
        max_errors = []
        
        for t in range(1, len(predicted_trajectories)-1):
            # Get valid points (non-nan)
            pred_valid = predicted_trajectories[t]
            true_valid = true_trajectories[t]
            valid_mask = ~np.isnan(pred_valid).any(axis=1) & ~np.isnan(true_valid).any(axis=1)
            
            if np.sum(valid_mask) > 0:
                pred_valid = pred_valid[valid_mask]
                true_valid = true_valid[valid_mask]
                
                # Calculate frame metrics
                mee, mee_std = self.mean_endpoint_error(pred_valid, true_valid)
                tlr, tlr_std = self.tracking_loss_rate(pred_valid, true_valid)
                max_err = self.maximum_endpoint_error(pred_valid, true_valid)
                
                mee_values.append(mee)
                tlr_values.append(tlr)
                max_errors.append(max_err)
        
        # Calculate temporal consistency
        temp_cons, temp_cons_std = self.temporal_consistency(predicted_trajectories)
        
        # Compile all metrics
        metrics['mean_endpoint_error'] = {
            'mean': np.mean(mee_values),
            'std': np.std(mee_values)
        }
        metrics['tracking_loss_rate'] = {
            'mean': np.mean(tlr_values),
            'std': np.std(tlr_values)
        }
        metrics['maximum_endpoint_error'] = {
            'value': np.max(max_errors)
        }
        metrics['temporal_consistency'] = {
            'mean': temp_cons,
            'std': temp_cons_std
        }
        
        return metrics

def print_metrics_table(metrics: dict):
    """
    Print metrics in a formatted table.
    """
    print("\nOptical Flow Performance Analysis")
    print("-" * 50)
    print(f"{'Metric':<30} {'Value':<10} {'Std Dev':<10}")
    print("-" * 50)
    print(f"Mean Endpoint Error (pixels)    {metrics['mean_endpoint_error']['mean']:8.2f}  ±{metrics['mean_endpoint_error']['std']:8.2f}")
    print(f"Maximum Endpoint Error (pixels) {metrics['maximum_endpoint_error']['value']:8.2f}  {'N/A':>8}")
    print(f"Tracking Loss Rate (%)          {metrics['tracking_loss_rate']['mean']:8.2f}  ±{metrics['tracking_loss_rate']['std']:8.2f}")
    print(f"Temporal Consistency            {metrics['temporal_consistency']['mean']:8.2f}  ±{metrics['temporal_consistency']['std']:8.2f}")
    print("-" * 50)

# Example usage:
if __name__ == "__main__":
    # Example data structure:
    # predicted_trajectories and true_trajectories should be numpy arrays of shape (T, N, 2)
    # where T is number of frames, N is number of points, and 2 is (x,y) coordinates
    
    # Initialize metrics calculator
    metrics_calculator = OpticalFlowMetrics(threshold_pixels=5.0)
    
    # Generate some dummy data for demonstration
    T, N = 100, 50  # 100 frames, 50 points
    true_trajectories = np.random.rand(T, N, 2) * 100  # Random positions in 100x100 space
    predicted_trajectories = true_trajectories + np.random.normal(0, 2, (T, N, 2))  # Add some noise
    
    # Calculate all metrics
    metrics = metrics_calculator.calculate_all_metrics(predicted_trajectories, true_trajectories)
    
    # Print results
    print_metrics_table(metrics) 