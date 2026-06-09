import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import json

from utils.synthetic_x_ray_dots import generate_xray_dataset
from training.train_model import (
    train_model, DotDataset, test_model, 
    visualize_filter_responses, visualize_network_progression, plot_training_metrics
) 
from model.dot_cnn import UNet
from inference.inference import process_all_frames


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    # Create runs directory if it doesn't exist
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Ask user for mode
    while True:
        mode_input = input("Choose mode:\n1. Pretrain on synthetic data\n2. Finetune on real data\n3. Run inference on calibration images\nEnter (1/2/3): ").strip()
        if mode_input in ['1', '2', '3']:
            mode = int(mode_input)
            break
        print("Please enter '1', '2', or '3'")
    
    ######################################################### Finetuning ######################################################### 
    ''' Here change the model to finetune on, e.g. "run_280925_1505" :) '''

    if mode == 2:
        # Call finetune.py script with specific model
        print("\nStarting finetuning process...")
        model_run = "run_280925_1505"  # Specify which model to finetune
        pretrained_path = os.path.join(runs_dir, model_run, "model.pt")
        from training import finetune
        finetune.main(pretrained_path=pretrained_path)
        sys.exit(0)
    ######################################################### Finetuning ######################################################### 
    
    ######################################################### Inference ######################################################### 
    elif mode == 3:
        # Run inference directly on calibration frames
        print("\nRunning inference on calibration frames...")
        finetuned_model_path = os.path.join(runs_dir, "run_280925_1505", "finetuned_model.pt")
        process_all_frames(
            frames_dir="data/frames",  # Use raw frames directory
            model_path=finetuned_model_path,
            threshold=0.5,  # Threshold for dot center extraction from predicted gaussian probability map
            num_frames=None,  # Process all frames
            run_dir=os.path.join(runs_dir, "inference_finetuned")  # Save results in separate directory
        )
        print("Inference complete!")
        sys.exit(0)
    ######################################################### Inference ######################################################### 

    ######################################################### Pretraining ######################################################### 
    else:  # mode == 1: Pretraining
        # Generate run name with timestamp
        timestamp = datetime.now().strftime("%d%m%y_%H%M")
        run_name = f"run_{timestamp}"
        run_dir = os.path.join(runs_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Generate X-ray dot detection dataset")
        parser.add_argument("--num-images", type=int, default=100, help="Number of images to generate")
        parser.add_argument("--image-size", type=int, default=256, help="Size of square images")
        parser.add_argument("--min-dots", type=int, default=50, help="Minimum number of dots per image")
        parser.add_argument("--max-dots", type=int, default=150, help="Maximum number of dots per image")
        parser.add_argument("--dot-radius", type=int, default=5, help="Radius of each dot in pixels")
        parser.add_argument("--output", default="data/synthetic_xray_dataset", help="Output directory")
        args = parser.parse_args()

        # Create config
        config = {
            'timestamp': timestamp,
            'training_mode': 'pretrain',
            'dataset_params': {
                'num_images': args.num_images,
                'image_size': args.image_size,
                'min_dots': args.min_dots,
                'max_dots': args.max_dots,
                'dot_radius': args.dot_radius,
                'dataset_dir': args.output
            },
            'device': str(get_device()),
            'training': {
                'perform_training': True
            }
        }

        # Save initial config
        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Generate synthetic X-ray dot dataset to train UNet model
        generate_xray_dataset(
            num_images=args.num_images,
            image_size=args.image_size,
            min_dots=args.min_dots,
            max_dots=args.max_dots,
            dot_radius=args.dot_radius,
            output_dir=args.output
        )
        print("Dataset generation complete!")
        
        # Train the model
        dataset_dir = args.output  # Use the same directory where we generated the dataset
        device = get_device()
        print(f"Using device: {device}")

        # Create dataset and dataloader with is_finetuning=False for pretraining
        dataset = DotDataset(dataset_dir, is_finetuning=False)
        train_size = int(0.8 * len(dataset))  # 80% for training, 20% for validation
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # Ask for number of epochs
        num_epochs = 10  # default value
        while True:
            try:
                epochs_input = input("Enter number of epochs to train (default=10): ").strip()
                if epochs_input == "":  # Use default if empty
                    break
                num_epochs = int(epochs_input)
                if num_epochs > 0:
                    break
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
        
        # Update config with epochs
        config['training'].update({
            'num_epochs': num_epochs
        })

        # Single training run
        model = UNet()
        model, metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device
        )

        # After training, update config with final metrics and save again
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

        # Save model and metrics
        model_save_path = os.path.join(run_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': config  # Also include config in model file for completeness
        }, model_save_path)
        print(f"Model and metrics saved in: {run_dir}")

        # Test model and visualize results
        print("Testing model and visualizing results...")
        test_model(model, val_dataset, device=device, save_dir=run_dir)
        
        # Save visualizations in the run directory
        print("Visualizing network progression...")
        sample_image, _ = dataset[0]
        visualize_network_progression(model, sample_image, device, save_dir=os.path.join(run_dir, 'visualizations'))
        
        # Visualize filter responses
        print("Visualizing filter responses...")
        visualize_filter_responses(model, sample_image, device, save_dir=os.path.join(run_dir, 'visualizations'))

    ######################################################### Pretraining ######################################################### 