# Dot Detection Pipeline

This repository contains a machine learning pipeline for detecting and localizing dots in X-ray images. The pipeline uses a U-Net architecture and consists of three main stages: pretraining on synthetic data, finetuning on real data, and inference.

## Project Structure

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

## Getting Started

The pipeline can be run in three different modes:
1. Pretraining on synthetic data
2. Finetuning on real data
3. Running inference on calibration images

To start the pipeline, run:

```bash
python detect_dots.py
```

You will be prompted to choose one of the three modes.

## 1. Pretraining Mode

Pretraining mode generates synthetic X-ray dot images and trains a U-Net model from scratch.

### Configuration Options:
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

## 2. Finetuning Mode

Finetuning adapts a pretrained model to real X-ray images. The process will:
1. Load a pretrained model (default: "run_280925_1505/model.pt")
2. Generate finetuning labels if not present
3. Train on real data with reduced learning rate
4. Save the finetuned model and metrics

### Key Features:
- Automatically generates finetuning labels if not present
- Uses a reduced batch size for large images
- Lower learning rate (0.0001) for stable finetuning
- 80/20 train/validation split
- Saves finetuned model in both original and new run directories

## 3. Inference Mode

Inference mode processes calibration frames using a trained model.

### Configuration:
- Input directory: `data/frames/`
- Default threshold: 0.5 (for dot center extraction)
- Output directory: `runs/inference_finetuned/`

## Runs Directory Structure

The `runs/` directory contains all training runs and their artifacts. Each run follows this naming convention:

```
runs/
├── run_DDMMYY_HHMM/           # Pretraining run
│   ├── config.json           # Run configuration
│   ├── model.pt             # Trained model checkpoint
│   ├── metrics.png          # Training metrics plot
│   ├── test_results/        # Model test visualizations
│   └── visualizations/      # Filter response visualizations
│
├── run_DDMMYY_HHMM_finetuned/ # Finetuning run
│   ├── config.json          # Finetuning configuration
│   ├── finetuned_model.pt   # Finetuned model checkpoint
│   └── metrics.png          # Finetuning metrics plot
│
└── inference_finetuned/     # Inference results
    └── calibration_grid_predictions/ # Predicted dot coordinates
```

### Key Files:
- `config.json`: Contains all run parameters and final metrics
- `model.pt`: Model checkpoint with state dict and training metrics
- `metrics.png`: Plot of loss and accuracy during training
- `visualizations/`: Network filter responses and progression visualizations

## Model Architecture

The pipeline uses a U-Net architecture (defined in `model/dot_cnn.py`) optimized for dot detection:
- Input: Grayscale X-ray images
- Output: Probability maps for dot locations
- Architecture: Encoder-decoder with skip connections

## Dependencies

Main dependencies:
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Best Practices

1. **Pretraining**:
   - Use a large synthetic dataset (1000+ images)
   - Monitor validation loss for overfitting
   - Check filter visualizations for learned features

2. **Finetuning**:
   - Use the latest stable pretrained model
   - Start with 20 epochs (adjust based on validation metrics)
   - Verify generated finetuning labels

3. **Inference**:
   - Use finetuned models for best results
   - Adjust threshold based on application needs
   - Monitor prediction quality on test frames
