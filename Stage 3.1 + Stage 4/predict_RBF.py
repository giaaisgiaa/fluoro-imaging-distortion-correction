import numpy as np
from utils.dataset_read import load_frame_datasets
import matplotlib.pyplot as plt
import os
from non_stationary_models.forward_rbf import SequentialRBF


def evaluate_model(target_frame=7, validation_split=0.2):
    """
    Evaluate the sequential RBF model on a specific target frame.
    """
    print("="*60)
    print(f"SEQUENTIAL RBF PREDICTION vs REAL COORDINATES")
    print(f"Target frame: {target_frame}")
    print(f"Validation split: {validation_split*100:.1f}%")
    print("="*60)
    
    # Load data
    datasets = load_frame_datasets("data/dot_trajectories.json")
    frames = sorted(datasets.keys(), key=lambda k: int(k.split()[-1]))
    data = np.stack([datasets[frame] for frame in frames], axis=0)
    
    # Split dots into training/validation
    n_dots = data.shape[1]
    n_validation = int(n_dots * validation_split)
    
    np.random.seed(42)
    all_indices = np.arange(n_dots)
    validation_indices = np.random.choice(all_indices, size=n_validation, replace=False)
    training_indices = np.setdiff1d(all_indices, validation_indices)
    
    training_data = data[:, training_indices, :]
    validation_data = data[:, validation_indices, :]
    
    # Create and train model
    model = SequentialRBF(
        kernel="thin_plate_spline",
        neighbors=80,
        smoothing=1e-3,
        time_weight=1.0,
        standardize=True
    )
    
    model.fit(training_data, t0=0.0, dt=1.0)
    
    # Get target frame data
    frame_idx = target_frame - 1
    if frame_idx >= len(data):
        print(f"Frame {target_frame} not available. Max frame: {len(data)}")
        return
        
    # Get previous frame positions
    prev_training = training_data[frame_idx - 1]
    prev_validation = validation_data[frame_idx - 1]
    
    # Get target frame positions (ground truth)
    target_training = training_data[frame_idx]
    target_validation = validation_data[frame_idx]
    
    # Make predictions
    training_predictions = model.predict(prev_training, frame_idx - 1)
    validation_predictions = model.predict(prev_validation, frame_idx - 1)
    
    # Compute errors
    training_squared_errors = np.sum((training_predictions - target_training) ** 2, axis=1)
    validation_squared_errors = np.sum((validation_predictions - target_validation) ** 2, axis=1)
    
    training_rmse = np.sqrt(np.mean(training_squared_errors))
    training_rmse_std = np.std(np.sqrt(training_squared_errors))
    training_max_error = np.max(np.sqrt(training_squared_errors))
    
    validation_rmse = np.sqrt(np.mean(validation_squared_errors))
    validation_rmse_std = np.std(np.sqrt(validation_squared_errors))
    validation_max_error = np.max(np.sqrt(validation_squared_errors))
    
    # Print results
    print("\nPREDICTION PERFORMANCE:")
    print("-"*40)
    print(f"Training dots:")
    print(f"  RMSE: {training_rmse:.3f} ± {training_rmse_std:.3f}")
    print(f"  Max error: {training_max_error:.3f}")
    print(f"\nValidation dots:")
    print(f"  RMSE: {validation_rmse:.3f} ± {validation_rmse_std:.3f}")
    print(f"  Max error: {validation_max_error:.3f}")
    
    return {
        'model': model,
        'training_rmse': training_rmse,
        'validation_rmse': validation_rmse,
        'training_predictions': training_predictions,
        'validation_predictions': validation_predictions,
        'training_data': training_data,
        'validation_data': validation_data
    }


def compute_average_performance(model, training_data, validation_data):
    """
    Compute average RMSE across all frames using an already trained model.
    """
    print("="*60)
    print("COMPUTING AVERAGE PERFORMANCE ACROSS ALL FRAMES")
    print("="*60)
    
    training_rmse_scores = []
    validation_rmse_scores = []
    frame_numbers = []
    
    print("Computing performance for each frame...")
    for target_frame in range(2, len(training_data)):  # Start from frame 2 (predict frame 1→2)
        # Get positions
        prev_training = training_data[target_frame - 1]  # Previous frame, at time t-1
        prev_validation = validation_data[target_frame - 1]  # Previous frame, at time t-1
        target_training = training_data[target_frame]  # Target frame, at time t
        target_validation = validation_data[target_frame]  # Target frame, at time t
        
        # Make predictions
        training_predictions = model.predict(prev_training, target_frame - 1)
        validation_predictions = model.predict(prev_validation, target_frame - 1)
        
        # Compute RMSE
        training_rmse = np.sqrt(np.mean(np.sum((training_predictions - target_training) ** 2, axis=1)))
        validation_rmse = np.sqrt(np.mean(np.sum((validation_predictions - target_validation) ** 2, axis=1)))
        
        training_rmse_scores.append(training_rmse)
        validation_rmse_scores.append(validation_rmse)
        frame_numbers.append(target_frame)
        
        if target_frame % 10 == 0:
            print(f"  Processed frame {target_frame}/{len(training_data)-1}")
    
    # Convert to numpy arrays
    training_rmse_scores = np.array(training_rmse_scores)
    validation_rmse_scores = np.array(validation_rmse_scores)
    
    # Calculate statistics
    training_mean = np.mean(training_rmse_scores)
    training_std = np.std(training_rmse_scores)
    validation_mean = np.mean(validation_rmse_scores)
    validation_std = np.std(validation_rmse_scores)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    time = frame_numbers
    
    plt.plot(time, validation_rmse_scores, 'b-', label='Validation RMSE', linewidth=2)
    plt.fill_between(time, 
                    validation_rmse_scores - validation_std,
                    validation_rmse_scores + validation_std,
                    color='b', alpha=0.2,
                    label='+/- std dev')
    
    plt.axhline(y=validation_mean, color='r', linestyle='--',
                label=f'Average RMSE = {validation_mean:.3f} ± {validation_std:.3f}',
                linewidth=2)
    
    plt.xlabel("Frame")
    plt.ylabel("RMSE (pixels)")
    plt.title("Validation RMSE over time (Sequential RBF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("visualizations/single_rbf", exist_ok=True)
    plot_filename = "visualizations/rbf/rmse_over_time_sequential.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved RMSE plot to: {plot_filename}")
    
    # Print final results
    print("\nFINAL PERFORMANCE METRICS")
    print("="*60)
    print(f"Training RMSE: {training_mean:.3f} ± {training_std:.3f} pixels")
    print(f"  Range: [{np.min(training_rmse_scores):.3f}, {np.max(training_rmse_scores):.3f}]")
    print(f"  25-75th percentile: [{np.percentile(training_rmse_scores, 25):.3f}, {np.percentile(training_rmse_scores, 75):.3f}]")
    
    print(f"\nValidation RMSE: {validation_mean:.3f} ± {validation_std:.3f} pixels")
    print(f"  Range: [{np.min(validation_rmse_scores):.3f}, {np.max(validation_rmse_scores):.3f}]")
    print(f"  25-75th percentile: [{np.percentile(validation_rmse_scores, 25):.3f}, {np.percentile(validation_rmse_scores, 75):.3f}]")
    
    print(f"\nFrames analyzed: {len(validation_rmse_scores)}")
    
    return {
        'training_rmse_mean': training_mean,
        'training_rmse_std': training_std,
        'validation_rmse_mean': validation_mean,
        'validation_rmse_std': validation_std,
        'training_rmse_scores': training_rmse_scores,
        'validation_rmse_scores': validation_rmse_scores,
        'frame_numbers': frame_numbers
    }


if __name__ == "__main__":
    # Get user input
    target_frame = input("Enter target frame number (default: 7): ").strip()
    target_frame = int(target_frame) if target_frame else 7
    
    validation_split = input("Enter validation split (0-1, default: 0.2): ").strip()
    validation_split = float(validation_split) if validation_split else 0.2
    
    # First evaluate on specific frame
    result = evaluate_model(target_frame, validation_split)
    
    # Then compute average performance across all frames
    print("\nComputing average performance across all frames...")
    avg_performance = compute_average_performance(
        result['model'],
        result['training_data'],
        result['validation_data']
    )