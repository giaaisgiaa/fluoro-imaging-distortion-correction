## Stage 3.1: Motion Prediction with Regression Models

This third stage focuses on predicting the motion of calibration grid points over time using different Machine Learning Regression models. The prediction task is formulated as:

Input features: X = (xt, yt, t)
- xt, yt: current position coordinates at time t
- t: current time step

Output: ut = (dx, dy) = (xt+1 - xt, yt+1 - yt)
- Displacement vector to next frame
- Predicts how much each point moves in one time step
- Next position is (xt+1, yt+1) = (xt + dx, yt + dy)

Each model learns to map X → ut, predicting one-step-forward motion.

### Implemented Models

1. **Linear Model** (`forward_linear.py`)
2. **Polynomial Model** (`forward_polynomial.py`)
3. **RBF (Radial Basis Function) Models**
   - Single RBF (`forward_rbf.py`):
     * Uses local RBF interpolation with thin plate spline kernel
     * Key parameters:
       - neighbors=80 (local interpolation using 80 nearest points)
       - smoothing=1e-3 (regularization for numerical stability)
       - time_weight=1.0 (temporal feature scaling)
     * Features are standardized before fitting
     * Predicts displacements using scipy.interpolate.RBFInterpolator
   - Ensemble RBF (`forward_rbf_ensemble.py`)
4. **XGBoost Model** (`forward_xgboost.py`)

### Running Models and Validation

To run any model, use its corresponding prediction script. For example, to run the Ensemble RBF model (best performing):

```bash
python predict_ensemble_RBF.py
```

The script will:
1. Load dot trajectories from `data/dot_trajectories.json`
2. Ask for validation split percentage (default 20%)
3. Train the model on 80% of the dots and validate on the remaining 20%
4. Generate RMSE validation plots in `visualizations/ensemble_rbf/rmse_over_time.png`

The validation split is done randomly but with a fixed seed (42) for reproducibility. The script will print:
- Training progress
- Validation RMSE for specific frames
- Overall model performance metrics

You can find similar RMSE validation plots for each model in their respective directories:
```
visualizations/
├── linear/rmse_over_time.png
├── polynomial/rmse_over_time_degree3.png
├── single_rbf/rmse_over_time.png
├── ensemble_rbf/rmse_over_time.png
└── xgboost/rmse_over_time.png
```

### Visualizing Predictions with Motion Overlay

To visualize how well the model predicts motion patterns, use:

```bash
python predict_motion_video_overlay.py --n_random_points 2000 --fps 30
```

Parameters:
- `--n_random_points`: Number of points to track (default: 2000)
- `--fps`: Frame rate of output video (default: 30)

This script will:
1. Train an Ensemble RBF model on the dot trajectories
2. Generate random points within the image bounds
3. Create a video showing:
   - Original grid points (ground truth)
   - Predicted motion paths
   - Overlay of predictions on actual movement
4. Save the visualization video for analysis

## Stage 4: Distortion Correction

This stage uses the trained models (primarily Ensemble RBF) to correct distortions in X-ray images using inverse pixel mapping techniques. The key insight is using forward motion prediction (Stage 3) to enable stable inverse mapping for image correction.

### Correction Process

1. **Forward Map Generation**
   - The Ensemble RBF model predicts how each pixel moves from frame 0 to the current frame
   - This forward prediction is more natural and accurate as it follows the physical motion
   - For each frame t, we get maps (map_x, map_y) that tell us where each pixel from frame 0 ends up
   - These maps capture the cumulative distortion up to frame t

2. **Image Warping with Inverse Mapping**
   - While we predict motion forward, we correct distortion using inverse (pull-back) mapping
   - For each pixel (x,y) in the corrected output:
     * Use map_x[y,x], map_y[y,x] to find its source in the distorted frame
     * Pull the color/intensity from that source position
   - Benefits of inverse mapping:
     * Every output pixel actively "pulls" its value from the input image
     * No holes are created because every output pixel is filled
     * No need for hole-filling interpolation techniques
     * Bilinear interpolation is only used for sub-pixel precision when sampling the source
   - Contrast with forward (push) mapping:
     * Pixels pushed from source to destination can leave gaps
     * Multiple source pixels might map to same destination
     * Would require additional hole-filling algorithms
   - Implementation using cv2.remap:
     ```python
     corrected_frame = cv2.remap(distorted_frame, 
                                map_x, map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
     ```

3. **Resumable Processing**
   - Correction maps are saved for each frame
   - Process can be paused and resumed
   - Enables correction of long video sequences
   - Saved state includes:
     * Model configuration
     * Current positions
     * Forward maps

### Key Concepts

1. **Forward Prediction, Inverse Correction**
   - Stage 3 predicts motion forward in time (natural physics)
   - Stage 4 uses these predictions to pull pixels backward (stable correction)
   - All corrections reference frame 0 as the undistorted state

2. **Pull-back vs Push-forward**
   - Pull-back (used here): each output pixel pulls its value from input
   - Push-forward (avoided): input pixels push values to output
   - Pull-back ensures no holes or multiple mappings

### Running the Correction Process

To correct distorted frames using the trained model:

```bash
# Basic usage
python correct_frames_resumable.py --n_frames 20 --start_frame 0

# Resume from a previous state
python correct_frames_resumable.py --n_frames 20 --start_frame 20 --resume
```

Parameters:
- `--n_frames`: Number of frames to process (default: 20)
- `--start_frame`: First frame to process, 0-indexed (default: 0)
- `--resume`: Flag to resume from saved state

The script will:
1. Load or train the Ensemble RBF model
2. Process frames sequentially, for each frame:
   - Generate forward mapping coordinates
   - Apply inverse mapping using cv2.remap
   - Save correction maps and state
3. Output corrected frames and mapping data

### Output Structure

```
data/
├── corrected_frames/      # Corrected output images
├── correction_state/      # Saved state for resuming
│   ├── model_config.json  # Model parameters
│   └── positions_frame_*.npy  # Saved positions per frame
├── forward_remaps/        # Generated mapping coordinates
│   ├── map_x_frame_*.npy  # X-coordinate maps
│   └── map_y_frame_*.npy  # Y-coordinate maps
└── frames/               # Original input frames
```

The process is resumable, so you can:
1. Start with a small batch of frames
2. Check the results
3. Continue with more frames using --resume
4. Process long videos in manageable chunks
