import os
import numpy as np
import cv2
from utils.dataset_read import load_frame_datasets
from non_stationary_models.forward_rbf_ensemble import ForwardRBFEnsemble
import glob
from pathlib import Path
from tqdm import tqdm
import json

def correct_frames(n_frames=20, start_frame=0, resume=False):
    """
    Correct distortion in original frames using ensemble RBF model.
    Processes frames one by one, saving state after each frame for resuming.
    
    Parameters:
    -----------
    n_frames : int
        Number of frames to process in this session (default: 20)
    start_frame : int
        Frame number to start from (0-indexed, default: 0)
    resume : bool
        Whether to resume from saved state (default: False)
    """
    print("="*60)
    print(f"CORRECTING FRAMES {start_frame} to {start_frame + n_frames - 1} USING ENSEMBLE RBF")
    print("="*60)
    
    # Create output directories
    output_dir = Path("data/corrected_frames")
    maps_dir = Path("data/forward_remaps")
    state_dir = Path("data/correction_state")
    
    output_dir.mkdir(exist_ok=True)
    maps_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    
    # Load datasets and train model (only if not resuming)
    if not resume or start_frame == 0:
        print("Loading datasets...")
        datasets = load_frame_datasets("data/dot_trajectories.json")
        
        # Convert to tensor format
        print("Converting data to tensor format...")
        frames = sorted(datasets.keys(), key=lambda k: int(k.split()[-1]))
        data = np.stack([datasets[frame] for frame in frames], axis=0)
        print(f"Data shape: {data.shape}, dtype: {data.dtype}")
        
        # Set random seed for consistent ensemble composition
        np.random.seed(42)
        
        # Train the Ensemble RBF model in hybrid mode
        print("Training Ensemble RBF model in hybrid mode...")
        model = ForwardRBFEnsemble(
            n_models=5,
            kernel="thin_plate_spline",
            neighbors=80,
            smoothing=1e-3,
            time_weight=1.0,
            standardize=True
        ).fit_tensor(data, t0=0.0, dt=1.0)
        
        # Save model parameters for resuming
        model_state = {
            'n_models': 5,
            'kernel': 'thin_plate_spline',
            'neighbors': 80,
            'smoothing': 1e-3,
            'time_weight': 1.0,
            'standardize': True,
            'data_shape': data.shape,
            'random_seed': 42
        }
        with open(state_dir / 'model_config.json', 'w') as f:
            json.dump(model_state, f)
        
        # Initialize current positions
        frame_files = sorted(glob.glob("data/frames/*.png"), 
                            key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        if not frame_files:
            print("No frame images found in data/frames/ directory!")
            return
        
        # Get frame dimensions from first frame
        first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
        height, width = first_frame.shape
        print(f"Frame dimensions: {width}x{height}")
        
        # Create initial grid
        xs = np.linspace(0, width - 1, width)
        ys = np.linspace(0, height - 1, height)
        XX, YY = np.meshgrid(xs, ys)
        current_positions = np.column_stack([XX.ravel(), YY.ravel()])
        
    else:
        # Resume from saved state
        print("Resuming from saved state...")
        
        # Load model configuration
        with open(state_dir / 'model_config.json', 'r') as f:
            model_state = json.load(f)
        
        # Set random seed for consistent ensemble composition (CRITICAL!)
        np.random.seed(model_state['random_seed'])
        
        # Recreate model
        model = ForwardRBFEnsemble(
            n_models=model_state['n_models'],
            kernel=model_state['kernel'],
            neighbors=model_state['neighbors'],
            smoothing=model_state['smoothing'],
            time_weight=model_state['time_weight'],
            standardize=model_state['standardize']
        )
        
        # Load datasets and retrain (needed for model state)
        datasets = load_frame_datasets("data/dot_trajectories.json")
        frames = sorted(datasets.keys(), key=lambda k: int(k.split()[-1]))
        data = np.stack([datasets[frame] for frame in frames], axis=0)
        model.fit_tensor(data, t0=0.0, dt=1.0)
        
        # Load current positions from last saved state
        current_positions = np.load(state_dir / f'positions_frame_{start_frame-1:03d}.npy')
        
        # Get frame dimensions
        frame_files = sorted(glob.glob("data/frames/*.png"), 
                            key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
        height, width = first_frame.shape
        print(f"Frame dimensions: {width}x{height}")
        print(f"Resuming from frame {start_frame} with {current_positions.shape[0]} positions")
    
    # Load frame files for this session
    frame_files = sorted(glob.glob("data/frames/*.png"), 
                        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))[start_frame:start_frame + n_frames]
    
    if not frame_files:
        print(f"No frame images found for frames {start_frame} to {start_frame + n_frames - 1}!")
        return
    
    print(f"Processing {len(frame_files)} frames one by one...")
    
    # Process each frame one by one (no pre-calculation!)
    for frame_idx, frame_file in enumerate(tqdm(frame_files, desc="Correcting frames")):
        actual_frame_num = start_frame + frame_idx
        
        # Read current frames
        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        
        # Build forward remap for this frame
        map_x, map_y = model.build_forward_remap(
            width=width,
            height=height,
            t_end=float(actual_frame_num),
            dt=1.0,
            batch=200_000
        )
        
        # Save correction maps
        np.save(maps_dir / f'map_x_frame_{actual_frame_num:03d}.npy', map_x)
        np.save(maps_dir / f'map_y_frame_{actual_frame_num:03d}.npy', map_y)
        
        # Apply correction
        corrected_frame = model.warp_image_to_frame0(
            frame, 
            map_x, 
            map_y,
            border_value=0
        )
        
        # Save corrected frame
        output_path = output_dir / f"corrected_{Path(frame_file).name}"
        cv2.imwrite(str(output_path), corrected_frame)
        
        # Save current state for resuming (after processing this frame)
        np.save(state_dir / f'positions_frame_{actual_frame_num:03d}.npy', np.array([map_x, map_y]))


    
    print(f"\nDone! Processed frames {start_frame} to {start_frame + len(frame_files) - 1}")
    print(f"Corrected frames saved to: {output_dir}")
    print(f"Correction maps saved to: {maps_dir}")
    print(f"State saved to: {state_dir}")
    print(f"\nTo resume next session, use:")
    print(f"correct_frames(n_frames=50, start_frame={start_frame + len(frame_files)}, resume=True)")

def main():
    print("="*60)
    print("FRAME CORRECTION - 3-NIGHT SCHEDULE")
    print("="*60)
    print("Night 1: Frames 0-49   (50 frames)")
    print("Night 2: Frames 50-99  (50 frames)")  
    print("Night 3: Frames 100-149 (50 frames)")
    print("="*60)
    
    while True:
        try:
            night = input("\nWhich night do you want to run? (1, 2, or 3): ").strip()
            
            if night == "1":
                print("\nStarting Night 1: Frames 0-49")
                correct_frames(n_frames=50, start_frame=0, resume=False)
                break
            elif night == "2":
                print("\nStarting Night 2: Frames 50-99")
                correct_frames(n_frames=50, start_frame=50, resume=True)
                break
            elif night == "3":
                print("\nStarting Night 3: Frames 100-149")
                correct_frames(n_frames=50, start_frame=100, resume=True)
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
