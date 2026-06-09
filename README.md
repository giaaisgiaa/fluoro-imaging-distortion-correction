# Distortion Correction Pipeline — ETH Zurich MSc Thesis

> **Data availability**: The calibration X-ray images, videos, and derived trajectory data used in this project are proprietary data from the Laboratory for Movement and Biomechanics (LMB) at ETH Zurich and are not publicly distributed. Only the source code is available in this repository. To reproduce the results, please contact the [LMB lab](https://lmb.ethz.ch/) or refer to the original thesis.

## Overview

The tracking Dual Plane Fluoroscope (tDPF), pioneered at ETH Zurich's Laboratory for Movement Biomechanics (LMB) as the **first of its kind worldwide**, moves with the patient during walking and produces X-ray images with highly localized, non-stationary geometric distortions that evolve in both space and time. Correcting these distortions is essential for accurate 2D–3D image registration and reliable dynamic joint kinematics analysis.

This project formulates distortion correction as a **spatio-temporal prediction problem** — motion depends jointly on spatial coordinates (x, y) and time (t) — and builds a full ML pipeline to automatically detect, track, and predict the motion of ~2,000 calibration grid points, then uses those predictions to correct the distortion frame by frame.

The final pipeline achieves a correction accuracy of **~0.13 mm (RMSE < 0.2 mm)** — a **16× improvement** over the lab's existing polynomial-based calibration system (~2.16 mm) — enabling reliable 2D/3D image registration for biomechanical joint kinematics analysis.

### Pipeline Architecture

```
Raw fluoroscopic X-ray video (1664 × 1600 px)
        │
        ▼
[Stage 1]  U-Net dot detection          →  100% detection rate, ~1.2 px localization error
           synthetic pretraining + real-data finetuning on ~2,028 grid points
        │
        ▼
[Stage 2]  Nearest-neighbour tracking   →  full trajectories for all ~2,028 grid points
        │
        ▼
[Stage 3]  Motion prediction            →  Ensemble RBF (best): 0.60 ± 0.04 px RMSE
           5 models benchmarked (Linear, Polynomial, XGBoost, RBF, LSTM)
        │
        ▼
[Stage 4]  Inverse pixel mapping        →  distortion-corrected output frames
           cv2.remap over 1664 × 1600 = 2,662,400 pixels per frame
```

### Key Results

| Stage | Model | Validation RMSE |
|-------|-------|-----------------|
| 1 – Dot Detection | U-Net (pretrain → finetune) | 100% detection rate, ~1.2 px error |
| 3 – Motion Prediction | **Ensemble RBF Kernels** | **0.60 ± 0.04 px (~0.13 mm)** ✓ selected |
| 3 – Motion Prediction | Single RBF Kernel | 0.62 ± 0.04 px |
| 3 – Motion Prediction | XGBoost | 1.20 ± 0.71 px |
| 3 – Motion Prediction | LSTM (Stage 3.2) | 4.17 ± 0.34 px (~0.90 mm) |
| 3 – Motion Prediction | Polynomial Regression (degree 3) | 4.18 ± 2.12 px |
| 3 – Motion Prediction | Linear Regression | 7.54 ± 4.10 px |

### Tech Stack

Python · PyTorch · OpenCV · NumPy · XGBoost · Matplotlib

### Future Directions

- **Tracking**: replace nearest-neighbour matching with Kalman filters or YOLO-based object tracking for robustness under complex motion
- **Deep learning**: explore GRUs, spatio-temporal transformers, and scheduled sampling to close the gap with RBF on longer horizons
- **Generalisation**: incorporate fluoroscope motion parameters into the feature vector to avoid recalibration across machine configurations
- **Ideal-grid correction**: train a dedicated network to map from the reference frame to a synthetic perfect grid, removing residual systematic bias
- **Speed**: parallelise pixel-wise RBF inference to reduce the current ~10 min/frame correction time

---

## Stage 1: Dot Detection

The first stage is a machine learning pipeline using U-Net architecture for detecting and localizing dots in X-ray images. The pipeline uses a three-phase approach: pretraining on synthetic data, finetuning on real data, and inference on real unseen calibration data.

### Project Structure

```
Stage 1/
├── data/                      # Data directory
│   ├── cine_videos/          # Raw cine video files and outputs
│   ├── coordinates_finetuning/ # Finetuning coordinate labels
│   ├── finetuning_labels/    # Labels for finetuning (frames & heatmaps)
│   ├── frames/               # Individual frames for processing
│   └── synthetic_xray_dataset/ # Generated synthetic training data
├── model/                    # Model architecture definition
├── training/                 # Training scripts
├── inference/               # Inference pipeline
├── utils/                   # Utility functions
└── runs/                    # Training runs and model checkpoints
```

### Getting Started

The pipeline can be run in three different modes:
1. Pretraining on synthetic data
2. Finetuning on real data
3. Running inference on calibration images

To start the pipeline, run:

```bash
python detect_dots.py
```

You will be prompted to choose one of the three modes.

### 1. Pretraining Mode

Pretraining mode generates synthetic X-ray dot images and trains a U-Net model from scratch.

#### Configuration Options:
- `--num-images`: Number of synthetic images to generate (default: 100)
- `--image-size`: Size of square images (default: 256)
- `--min-dots`: Minimum dots per image (default: 50)
- `--max-dots`: Maximum dots per image (default: 150)
- `--dot-radius`: Radius of dots in pixels (default: 5)
- `--output`: Output directory for synthetic dataset (default: "data/synthetic_xray_dataset")

The pretraining process will:
1. Generate a synthetic dataset
2. Train the U-Net model
3. Save model checkpoints and visualizations
4. Generate training metrics and filter response visualizations

### 2. Finetuning Mode

Finetuning adapts a pretrained model to real X-ray images. The process will:
1. Load a pretrained model (default: "run_280925_1505/model.pt")
2. Generate finetuning labels if not present
3. Train on real data with reduced learning rate
4. Save the finetuned model and metrics

#### Key Features:
- Automatically generates finetuning labels if not present
- Uses a reduced batch size for large images
- Lower learning rate (0.0001) for stable finetuning
- 80/20 train/validation split
- Saves finetuned model in both original and new run directories

### 3. Inference Mode

Inference mode processes calibration frames using a trained model.

#### Configuration:
- Input directory: `data/frames/`
- Output directory: `runs/inference_finetuned/`

### Model Architecture

The pipeline uses a U-Net architecture (defined in `model/dot_cnn.py`) optimized for dot detection:
- Input: Grayscale X-ray images
- Output: Gaussian probability maps for dot locations
- Architecture: Encoder-decoder with skip connections

## Stage 2: Dot Follow

This second stage is a custom dot tracking system that follows individual dots across multiple frames using a Nearest Neighbor Search (NNS) approach. The system is designed to be memory-efficient and handle large numbers of dots across multiple frames.

### System Overview

The tracking system consists of several key components:

1. **Coordinate Cleaning** - Removes duplicate dots (rare occurrences). 
                          U-Net showed only in the very first frame (reference) 2 double dot detections, 
                          very close to each other
2. **Grid Labeling** - Labels each dot with a unique identifier
3. **Trajectory Tracking** - Follows dots across frames using custom nearest neighbor search algorithm and 
                          updates the dots indexes accordingly 
4. **Coordinate Storage** - Saves tracked trajectories in both NumPy and JSON formats

### Directory Structure

```
Stage 2/
├── data/
│   ├── coordinates/          # JSON coordinate files
│   ├── coordinates_numpy/    # NumPy coordinate files
│   ├── frames/              # Input frame images
│   ├── labels_raw/          # Initial dot labels
│   ├── labels_tracked/      # Tracked dot labels
│   └── trajectories/        # Final dot trajectories
├── utils/
│   ├── clean_dots.py        # Coordinate cleaning utilities
│   ├── data_loading.py      # Data loading functions
│   ├── follow_dot.py        # Core tracking implementation
│   └── visualization.py     # Visualization utilities
├── follow_dot.py            # Main execution script
└── dot_trajectories.json    # Final trajectory data
```

### Using the Scripts

#### 1. Dot Tracking (`follow_dot.py`)

This is the main script that processes the frames and tracks dots:

```bash
python follow_dot.py
```

The script will:
1. Count total frames in `data/frames/`
2. Clean coordinates to remove any duplicates
3. Run the dots labeling if needed
4. Compute and save trajectories
5. Generate `dot_trajectories.json`

Output files will be created in their respective directories under `data/`.

#### 2. Visualization (`visualize_trajectories.py`)

This script provides interactive visualization of the tracked dots:

```bash
python visualize_trajectories.py
```

The script offers two visualization modes:

1. **All Trajectories Mode** (Option 1):
   - Shows all dot trajectories overlaid on a single plot
   - Creates `all_trajectories.png`
   - Useful for overall movement pattern analysis
   - Shows start points (green) and end points (red)

2. **Selected Dots Mode** (Option 2):
   - Allows detailed analysis of specific dots
   - Interactive prompts for:
     ```
     Enter dot indices to track (space-separated numbers, e.g., '1 2 3'): 
     Enter number of frames to process (press Enter for default=150):
     Generate animation? (y/n, press Enter for yes):
     ```
   - Outputs:
     - Individual trajectory plots: `dot_X_trajectory.png`
     - Optional animation: `dots_X_Y_Z_animation.mp4`

### Output Files

After running both scripts, you'll have:

1. **Tracking Data**:
   - `data/labels_tracked/`: Tracked dot positions
   - `data/trajectories/`: Individual dot trajectories
   - `dot_trajectories.json`: Complete tracking data

2. **Visualizations**:
   - `all_trajectories.png`: Overview of all dot movements
   - `dot_X_trajectory.png`: Individual dot trajectory plots
   - `dots_X_Y_Z_animation.mp4`: Animated visualization (if enabled)

### Notes

- The visualization script requires the tracking script ('follow_dot.py') to be run first
- Animations require ffmpeg to be installed


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
   - Single RBF (`forward_rbf.py`)
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

## Stage 3.2: Motion Prediction with LSTM

An alternative approach using LSTM neural networks for predicting dot trajectories.

### Pipeline Steps and Execution

1. **Data Transformation** (`transform_to_tracks.py`):
   ```bash
   python transform_to_tracks.py
   ```
   - Converts raw dot trajectories from `data/dot_trajectories.json` into processed tracks
   - Outputs `data/tracks.json` for model training
   - Generates the ML-ready dataset `data/dataset.npz`

2. **Training the LSTM Model** (`train_lstm_with_anchors.py`):
   ```bash
   python train_lstm_with_anchors.py
   ```
   - Trains the LSTM model on the prepared dataset
   - Creates a new run directory in `runs/run-TIMESTAMP/`

3. **Visualization** (`viz_lstm_rollout.py`):
   ```bash
   python viz_lstm_rollout.py --run_dir runs/run-TIMESTAMP
   ```
   - Generates visualization of model predictions in `runs/run-TIMESTAMP/viz/`

### Directory Structure

```
.
├── data/
│   ├── dataset.npz          # ML-ready dataset
│   ├── dot_trajectories.json # Raw input data
│   └── tracks.json          # Processed trajectory data
├── runs/
│   └── run-TIMESTAMP/       # Training run artifacts
│       ├── config.json
│       ├── model.pt
│       ├── norm_mean.npy
│       ├── norm_std.npy
│       ├── splits.json
│       └── viz/             # Visualization outputs
└── *.py                     # Pipeline scripts
```

### LSTM Model Architecture

The model uses the following vector structure:

**Input features X = (t, δt, xt, yt, x0, y0) ∈ ℝ⁶**
- t: Current timestamp
- δt: Time delta since sequence start
- (xt, yt): Current dot position
- (x0, y0): Anchor point (initial position)

**Target y = (xt+1, yt+1) ∈ ℝ²**
- Next timestamp dot position

### Training Process

1. **Data Preparation**:
   - Trajectories are normalized using mean and standard deviation
   - Data is split into training (80%) and validation (20%) sets
   - Sequences are created with appropriate sliding windows

2. **Model Training**:
   - Loss function: Mean Squared Error (MSE)
   - Optimization: Adam optimizer with learning rate scheduling
   - Early stopping is implemented to prevent overfitting

### Performance

The LSTM model achieved submillimeter accuracy on the validation set, 
but showed lower performance compared to the ensemble RBF kernel regression model from Stage 3.1. 
Therefore, the RBF model was selected for implementation in Stage 4: Distortion Correction.


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



# Optical Flow (Experimental Approach)

An experimental approach using optical flow for distortion correction in X-ray images, which ultimately proved unsuitable for this complex distortion correction task.

## Technical Approach

### Feature Detection and Tracking

The pipeline uses a two-stage approach for tracking features across frames:

1. **Feature Extraction (First Frame Only)**
   - Utilizes the Shi-Tomasi corner detector (also known as Good Features to Track)
   - Features are extracted only from the first frame to ensure consistent tracking
   - The algorithm identifies corners by analyzing intensity gradients in local neighborhoods
   - These features ideally correspond to the calibration grid dots

2. **Feature Tracking (Subsequent Frames)**
   - The Lucas-Kanade optical flow algorithm tracks the initially detected features
   - Tracks features frame-by-frame by solving optical flow equations
   - Assumes small motion between consecutive frames
   - Attempts to follow the same physical points (dots) throughout the sequence

### Why This Approach Was Chosen

- Extracting features only in the first frame should theoretically allow for more reliable tracking of the same physical points over time
- The Lucas-Kanade algorithm is well-suited for tracking sparse feature sets
- The approach aimed to understand and correct underlying motion patterns of the dots

## Limitations and Failure Modes

The approach proved unsuccessful for several reasons:

1. **Feature Loss**
   - High percentage of features were lost during tracking
   - Complex distortion patterns caused features to become untrackable

2. **Motion Complexity**
   - Sudden changes in the first derivative of motion
   - Distortion patterns were too complex for the optical flow assumptions

3. **Feature Ambiguity**
   - Large number of dots (2028) with similar intensity values
   - Difficult to maintain reliable feature correspondence

## Setup Requirements

1. Create folder "data" including subfolders:
   - coordinates (JSON files, dots coordinates from Stage 1)
   - frames (original X-Ray frames of calibration grid)
   - frames_prepro (preprocessed calibration frames, if exist!)

## Available Methods

1. Original X-ray images with Shi-Tomasi corner detection
2. Preprocessed binary images with Shi-Tomasi corner detection
3. Use extracted dot centers (from JSON) as features

## Conclusion

ATTENTION: Due to the complex nature of the distortion problem in the dual Plane Tracking Fluoroscope, the optical flow approach was not able to reliably detect the motion patterns and was excluded as a viable method for distortion correction. The combination of feature loss, complex motion patterns, and feature ambiguity made it impossible to achieve reliable tracking results.

### Overall ML_Pipeline Dependencies

Main dependencies:
- PyTorch
- OpenCV
- NumPy
- Matplotlib
