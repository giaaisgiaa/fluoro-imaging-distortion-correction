import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import json
from training.train_model import DotDataset, train_model, plot_training_metrics, test_model
from model.dot_cnn import UNet
import sys
import subprocess

def load_pretrained_model(model_path, device):
    """Load pretrained model and its configuration"""
    print(f"Loading pretrained model from: {model_path}")
    
    model = UNet()
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Get original config if available
        config = checkpoint.get('config', None)
    else:
        model.load_state_dict(checkpoint)
        config = None
    
    return model, config

def main(pretrained_path=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Check if finetuning data exists
    dataset_dir = os.path.join("data", "finetuning_labels")
    frames_dir = os.path.join(dataset_dir, "frames")
    heatmaps_dir = os.path.join(dataset_dir, "heatmaps")
    
    if not os.path.exists(frames_dir) or not os.path.exists(heatmaps_dir):
        print(f"\nFinetuning data not found in {dataset_dir}")
        print("Running create_finetuning_label.py to generate the dataset...")
        
        # Get the absolute path to create_finetuning_label.py
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        create_labels_script = os.path.join(script_dir, "utils", "create_finetuning_label.py")
        
        try:
            # Run the script
            subprocess.run([sys.executable, create_labels_script], check=True)
            print("\nFinetuning data generated successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\nError generating finetuning data: {e}")
            sys.exit(1)
        
        # Verify data was created
        if not os.path.exists(frames_dir) or not os.path.exists(heatmaps_dir):
            print("\nError: Finetuning data still not found after running generation script")
        sys.exit(1)
    
    # Check if we have any frames to finetune on
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    if not frame_files:
        print(f"Error: No frames found in {frames_dir}")
        print("Please ensure finetuning data is properly generated.")
        sys.exit(1)
    
    print(f"Found {len(frame_files)} frames for finetuning")
    
    # Create runs directory if it doesn't exist
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    # Use provided pretrained path or default
    if pretrained_path is None:
        pretrained_path = os.path.join(runs_dir, "run_280925_1505", "model.pt")
    
    # Load pretrained model
    model, pretrained_config = load_pretrained_model(pretrained_path, device)
    
    # Extract timestamp from pretrained model path
    model_dir = os.path.dirname(pretrained_path)
    timestamp = os.path.basename(model_dir).split('_')[1]  # Gets "280925" from "run_280925_1505"
    time = os.path.basename(model_dir).split('_')[2]  # Gets "1505" from "run_280925_1505"
    
    # Use same timestamp for finetuned run
    run_name = f"run_{timestamp}_{time}_finetuned"
    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create config for finetuning
    config = {
        'timestamp': f"{timestamp}_{time}",
        'pretrained_model': pretrained_path,
        'device': str(device),
        'training': {
            'perform_training': True,
            'batch_size': 1,  # Reduced batch size for large images
            'optimizer': {
                'name': 'Adam',
                'lr': 0.0001  # Lower learning rate for finetuning
            },
            'model': {
                'type': 'UNet',
                'input_channels': 1,
                'output_channels': 1
            }
        }
    }
    
    # Add pretrained config if available
    if pretrained_config:
        config['pretrained_config'] = pretrained_config
    
    # Save initial config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load finetuning dataset, which DotDataset recognizes thanks to bool is_finetuning=True :)
    full_dataset = DotDataset(dataset_dir, is_finetuning=True)
    
    # Split dataset into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update config with dataset info
    config['training'].update({
        'data_split': {
            'train_size': train_size,
            'val_size': val_size,
            'train_ratio': 0.8
        },
        'dataset_dir': dataset_dir,
        'total_samples': len(full_dataset)
    })
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Ask for number of epochs
    while True:
        try:
            epochs_input = input("Enter number of epochs for finetuning (default=20): ").strip()
            num_epochs = 20 if epochs_input == "" else int(epochs_input)
            if num_epochs > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    config['training']['num_epochs'] = num_epochs
    
    # Finetune model
    print("\nStarting finetuning...")
    model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    # Update config with final metrics
    config['training'].update({
        'final_metrics': metrics,
        'completed_timestamp': datetime.now().strftime("%d%m%y_%H%M"),
        'training_duration': datetime.now().strftime("%H:%M:%S")
    })
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Plot training metrics
    plot_training_metrics(metrics, save_dir=run_dir)
    
    # Save finetuned model in both directories
    # 1. In original model directory
    pretrained_dir = os.path.dirname(pretrained_path)
    original_save_path = os.path.join(pretrained_dir, "finetuned_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': config
    }, original_save_path)
    print(f"Finetuned model saved in original directory: {original_save_path}")
    
    # 2. In finetuned run directory
    finetuned_save_path = os.path.join(run_dir, "finetuned_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': config
    }, finetuned_save_path)
    print(f"Finetuned model saved in finetuned directory: {finetuned_save_path}")
    
    # Test model and visualize results
    print("\nTesting finetuned model and visualizing results...")
    test_model(model, val_dataset, device=device, save_dir=run_dir)
    
    print("\nFinetuning complete!")
    print(f"Results saved in: {run_dir}")