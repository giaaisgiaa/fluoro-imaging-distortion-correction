# LSTM-based Dot Trajectory Prediction

This workspace contains a machine learning pipeline for predicting dot trajectories using an LSTM neural network. The pipeline consists of three main steps:

## 1. Data Transformation
First, run the data transformation script to convert raw trajectory data into a suitable format:

```bash
python transform_to_tracks.py
```

This script:
- Takes raw dot trajectory data from `data/dot_trajectories.json`
- Converts it into a track-centric format saved as `data/tracks.json`
- Also generates an NPZ file (`data/dataset.npz`) optimized for machine learning

## 2. Training the LSTM Model
Train the LSTM model using:

```bash
python train_lstm_with_anchors.py
```

Key features:
- Uses a sliding window approach to predict next positions
- Includes anchor points (initial positions) as additional features
- Automatically creates a timestamped run directory under `runs/` containing:
  - `model.pt`: Trained model weights
  - `config.json`: Training configuration
  - `splits.json`: Train/validation split information
  - `norm_mean.npy` & `norm_std.npy`: Normalization statistics (if enabled)

Training parameters can be customized:
- `--window`: Sliding window size (default: 25)
- `--val_frac`: Validation set fraction (prompted during runtime)
- `--epochs`: Number of training epochs (default: 50)
- `--normalize`: Enable input normalization
- `--hidden`: Hidden layer size (default: 64)
- `--layers`: Number of LSTM layers (default: 1)

## 3. Visualization
Visualize the model's predictions using:

```bash
python viz_lstm_rollout.py
```

This script:
- Interactively lists available run folders for selection
- Creates visualizations under the run's `viz/` directory:
  - `rollout_val.mp4`: Animation showing ground truth vs. predictions
  - `rmse_plot.png`: Plot of prediction error over time

Visualization features:
- Green dots: Ground truth trajectories
- Orange dots: Model predictions
- Red lines: Per-point prediction errors

## Directory Structure
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
│       ├── splits.json
│       └── viz/             # Visualization outputs
└── *.py                     # Pipeline scripts
```
