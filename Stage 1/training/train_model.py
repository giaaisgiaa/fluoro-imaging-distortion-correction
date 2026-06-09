import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.error_metrics import calculate_detection_metrics, find_dot_centers
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Custom dataset for both synthetic and finetuning data
class DotDataset(Dataset):
    def __init__(self, dataset_dir, is_finetuning=False, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.is_finetuning = is_finetuning
        
        if self.is_finetuning:
            # For finetuning data
            frames_dir = os.path.join(dataset_dir, 'frames')
            if not os.path.exists(frames_dir):
                raise ValueError(f"Finetuning data directory structure not found in {dataset_dir}")
            self.image_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        else:
            # For synthetic data
            self.image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        if self.is_finetuning:
            # For finetuning data
            img_path = os.path.join(self.dataset_dir, 'frames', img_name)
            json_path = os.path.join(self.dataset_dir, 'heatmaps', img_name.replace('.png', '_meta.json'))
            heatmap_path = os.path.join(self.dataset_dir, 'heatmaps', img_name.replace('.png', '.npy'))
            
            # Load image and normalize
            image = Image.open(img_path).convert('L')
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Load pre-computed heatmap
            heatmap = np.load(heatmap_path)
            
        else:
            # For synthetic data
            img_path = os.path.join(self.dataset_dir, img_name)
            json_path = os.path.join(self.dataset_dir, img_name.replace('.png', '.json'))
            
            # Load image and normalize
            image = Image.open(img_path).convert('L')
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Load dot locations and create heatmap
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            image_size = data['image_size']
            dot_radius = data['dot_radius']
            
            # Create target heatmap (Gaussian blobs at dot centers)
            heatmap = np.zeros((image_size, image_size), dtype=np.float32)
            for dot in data['dots']:
                x, y = dot['x'], dot['y']
                # Create a small Gaussian blob at each dot center
                for i in range(max(0, x - dot_radius * 3), min(image_size, x + dot_radius * 3)):
                    for j in range(max(0, y - dot_radius * 3), min(image_size, y + dot_radius * 3)):
                        dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                        # Gaussian with sigma = dot_radius/2
                        heatmap[j, i] = max(heatmap[j, i], 
                                       np.exp(-dist**2 / (2 * (dot_radius/2)**2)))
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Add channel dimension
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # Add channel dimension
        
        return image_tensor, heatmap_tensor



def plot_training_metrics(metrics, save_dir='runs/latest'):
    """Plot training and validation metrics"""
    # Create visualizations subdirectory
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    train_epochs = [m['epoch'] for m in metrics['train']]
    train_losses = [m['loss'] for m in metrics['train']]
    train_rmse = [np.sqrt(m['loss']) for m in metrics['train']]  # Calculate RMSE
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot MSE Loss
    ax1.plot(train_epochs, train_losses, label='Training Loss', color='blue')
    if 'val' in metrics and metrics['val']:
        val_epochs = [m['epoch'] for m in metrics['val']]
        val_losses = [m['loss'] for m in metrics['val']]
        ax1.plot(val_epochs, val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Validation MSE')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RMSE
    ax2.plot(train_epochs, train_rmse, label='Training RMSE', color='blue')
    if 'val' in metrics and metrics['val']:
        val_rmse = [np.sqrt(m['loss']) for m in metrics['val']]
        ax2.plot(val_epochs, val_rmse, label='Validation RMSE', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Training and Validation RMSE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model, train_loader, val_loader=None, num_epochs=50, device=None):
    if device is None:
        device = get_device()
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on device: {device}")
    
    # Track metrics
    train_metrics = []
    val_metrics = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_metrics.append({'epoch': epoch + 1, 'loss': epoch_loss})
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * images.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_metrics.append({'epoch': epoch + 1, 'loss': val_loss})
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Training Loss: {epoch_loss:.4f}')
            if val_loader is not None:
                print(f'  Validation Loss: {val_loss:.4f}')
    
    # Restore best model if validation was used
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nRestored best model based on validation loss!")
    
    metrics = {'train': train_metrics, 'val': val_metrics if val_loader is not None else []}
    return model, metrics



def test_model(model, dataset, device=None, save_dir='runs/latest'):
    """Test model and visualize results with detailed metrics"""
    if device is None:
        device = get_device()
    
    model.eval()
    
    # Create directory for validation predictions
    val_pred_dir = os.path.join(save_dir, 'synthetic_validation_prediction')
    os.makedirs(val_pred_dir, exist_ok=True)
    
    # Lists to store metrics for all validation images
    all_distances = []
    all_detection_rates = []
    
    # Process all validation images for metrics
    for idx in range(len(dataset)):
        image, true_heatmap = dataset[idx]
        
        # Get true centers
        if hasattr(dataset, 'dataset'):
            original_dataset = dataset.dataset
            true_idx = dataset.indices[idx]
        else:
            original_dataset = dataset
            true_idx = idx
            
        # Get JSON path based on dataset type
        if original_dataset.is_finetuning:
            img_name = original_dataset.image_files[true_idx]
            json_path = os.path.join(original_dataset.dataset_dir, 'heatmaps', 
                                   img_name.replace('.png', '_meta.json'))
        else:
            img_name = original_dataset.image_files[true_idx]
            json_path = os.path.join(original_dataset.dataset_dir, 
                                   img_name.replace('.png', '.json'))
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get dots based on dataset type
        if original_dataset.is_finetuning:
            true_centers = np.array([(dot['x'], dot['y']) for dot in data['dots']])
        else:
            true_centers = np.array([(dot['x'], dot['y']) for dot in data['dots']])
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            pred_heatmap = model(image_tensor)
        
        # Get predicted centers
        pred_heatmap_np = pred_heatmap.squeeze().cpu().numpy()
        pred_centers = find_dot_centers(pred_heatmap_np, threshold=0.1)  # Lower threshold for initial training
        
        # Convert centers to numpy array for plotting
        if len(pred_centers) > 0:
            pred_centers_array = np.array([[center['x'], center['y']] for center in pred_centers])
        else:
            pred_centers_array = np.array([]).reshape(0, 2)  # Empty array with correct shape
        
        # Debug print
        print(f"\nImage {idx}:")
        print(f"Predicted centers shape: {pred_centers_array.shape}")
        print(f"True centers shape: {true_centers.shape}")
        print(f"Number of predicted centers: {len(pred_centers)}")
        print(f"Number of true centers: {len(true_centers)}")
        
        # Calculate metrics for this image
        detection_metrics = calculate_detection_metrics(pred_heatmap_np, true_centers, threshold=0.1)
        
        # Create visualization
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        
        # Input image with true centers
        ax1.imshow(image.squeeze().cpu(), cmap='gray')
        ax1.scatter(true_centers[:, 0], true_centers[:, 1], c='g', marker='+', s=100, label='True Centers')
        ax1.set_title('Input Image with True Centers')
        ax1.legend()
        ax1.axis('off')
        
        # Input image with predicted centers
        ax2.imshow(image.squeeze().cpu(), cmap='gray')
        if len(pred_centers_array) > 0:
            ax2.scatter(pred_centers_array[:, 0], pred_centers_array[:, 1], c='r', marker='x', s=100, label='Predicted Centers')
        ax2.set_title(f'Input Image with Predicted Centers ({len(pred_centers)} dots)')
        ax2.legend()
        ax2.axis('off')
        
        # Input image with both centers and connections
        ax3.imshow(image.squeeze().cpu(), cmap='gray')
        ax3.scatter(true_centers[:, 0], true_centers[:, 1], c='g', marker='+', s=100, label='True Centers')
        if len(pred_centers_array) > 0:
            ax3.scatter(pred_centers_array[:, 0], pred_centers_array[:, 1], c='r', marker='x', s=100, label='Predicted Centers')
            
            # Draw lines between matched centers
            distances = cdist(true_centers, pred_centers_array)
            true_idx, pred_idx = linear_sum_assignment(distances)
            matched_distances = distances[true_idx, pred_idx]
            
            for i, (t_idx, p_idx) in enumerate(zip(true_idx, pred_idx)):
                if matched_distances[i] <= 10:  # Only draw lines for matches within threshold
                    ax3.plot([true_centers[t_idx, 0], pred_centers_array[p_idx, 0]], 
                            [true_centers[t_idx, 1], pred_centers_array[p_idx, 1]], 
                            'y--', alpha=0.5)
        
        ax3.set_title('True (green +) vs Predicted (red x) Centers\nYellow lines show matches')
        ax3.legend()
        ax3.axis('off')
        
        # Heatmap prediction
        ax4.imshow(pred_heatmap.squeeze().cpu(), cmap='hot')
        ax4.set_title('Predicted Heatmap')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(val_pred_dir, f'validation_results_{idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store metrics
        if not np.isinf(detection_metrics['mean_distance']):
            all_distances.append(detection_metrics['mean_distance'])
        detection_rate = (detection_metrics['num_correct'] / detection_metrics['num_true']) * 100
        all_detection_rates.append(detection_rate)
    
    # Calculate average metrics across all validation images
    mean_distance = np.mean(all_distances)
    std_distance = np.std(all_distances)
    mean_detection_rate = np.mean(all_detection_rates)
    std_detection_rate = np.std(all_detection_rates)
    
    print("\nValidation Metrics (averaged across all validation images):")
    print(f"Mean Distance: {mean_distance:.2f} ± {std_distance:.2f} pixels")
    print(f"Detection Rate: {mean_detection_rate:.1f} ± {std_detection_rate:.1f}%")
    
    # Save metrics to JSON
    metrics_summary = {
        'mean_distance': float(mean_distance),
        'std_distance': float(std_distance),
        'mean_detection_rate': float(mean_detection_rate),
        'std_detection_rate': float(std_detection_rate)
    }
    
    with open(os.path.join(val_pred_dir, 'validation_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    return metrics_summary
 
def visualize_filter_responses(model, sample_image, device, save_dir='runs/latest/visualizations'):
    """Visualize how the first layer filters respond to an actual dot image"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Get the first conv layer
    first_conv = model.enc1[0]
    
    # Create figure
    plt.style.use('default')
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    # Show input image
    axes[0, 0].imshow(sample_image.squeeze().cpu(), cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Get filter responses
    with torch.no_grad():
        input_tensor = sample_image.unsqueeze(0).to(device)
        responses = first_conv(input_tensor)
    
    # Show first 7 filter responses (1 input image + 7 responses = 8 plots)
    for i in range(7):
        row = i // 4
        col = (i % 4) + (1 if i < 3 else 0)  # Shift by 1 in first row to account for input image
        response = responses[0, i].cpu()
        axes[row, col].imshow(response, cmap='viridis')
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'filter_responses.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_network_progression(model, sample_image, device, save_dir='runs/latest/visualizations'):
    """Visualize how the image progresses through the network"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Dictionary to store intermediate activations
    activations = {}
    
    # Hook function to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for each layer we want to visualize
    hooks = []
    hooks.append(model.enc1.register_forward_hook(get_activation('encoder1')))
    hooks.append(model.enc2.register_forward_hook(get_activation('encoder2')))
    hooks.append(model.enc3.register_forward_hook(get_activation('encoder3')))
    hooks.append(model.bottleneck.register_forward_hook(get_activation('bottleneck')))
    hooks.append(model.dec3.register_forward_hook(get_activation('decoder3')))
    hooks.append(model.dec2.register_forward_hook(get_activation('decoder2')))
    hooks.append(model.dec1.register_forward_hook(get_activation('decoder1')))
    
    # Forward pass with sample image
    with torch.no_grad():
        input_tensor = sample_image.unsqueeze(0).to(device)
        output = model(input_tensor)
    
    # Create visualization
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 8))
    
    def plot_feature_maps(activation, title, subplot_idx):
        plt.subplot(2, 4, subplot_idx)
        # Average across channels for visualization
        mean_activation = activation.mean(dim=1).cpu().squeeze()
        # Normalize for better visualization
        mean_activation = (mean_activation - mean_activation.min()) / (mean_activation.max() - mean_activation.min() + 1e-8)
        plt.imshow(mean_activation, cmap='viridis')
        plt.title(title, pad=20, fontsize=12)
        plt.axis('off')
    
    # Plot the progression through the network
    plot_feature_maps(activations['encoder1'], 'Encoder Block 1', 1)
    plot_feature_maps(activations['encoder2'], 'Encoder Block 2', 2)
    plot_feature_maps(activations['encoder3'], 'Encoder Block 3', 3)
    plot_feature_maps(activations['bottleneck'], 'Bottleneck', 4)
    plot_feature_maps(activations['decoder3'], 'Decoder Block 3', 5)
    plot_feature_maps(activations['decoder2'], 'Decoder Block 2', 6)
    plot_feature_maps(activations['decoder1'], 'Decoder Block 1', 7)
    
    # Plot final output
    plt.subplot(2, 4, 8)
    plt.imshow(output.squeeze().cpu(), cmap='hot')
    plt.title('Final Output', pad=20, fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'network_progression.png'), dpi=300, bbox_inches='tight')
    plt.close()
 
    # Remove hooks
    for hook in hooks:
        hook.remove() 