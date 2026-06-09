import cv2
import numpy as np
import os
import glob
import json
from matplotlib import pyplot as plt

def preprocess_xray(image):
    """
    Preprocess X-ray image for better corner detection.
    """
    # Convert to float32
    img_float = image.astype(np.float32)
    
    # Normalize
    img_norm = cv2.normalize(img_float, None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply((img_norm * 255).astype(np.uint8))
    
    # Denoise
    img_denoised = cv2.fastNlMeansDenoising(img_clahe)
    
    return img_denoised

def detect_corners(image, max_corners=3000, quality_level=0.01, min_distance=10):
    """
    Detect corners using Shi-Tomasi corner detector.
    """
    # Preprocess image
    processed = preprocess_xray(image)
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        processed,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )
    
    return corners

def visualize_corners(image, corners, output_path=None):
    """
    Visualize detected corners with bright yellow crosses.
    """
    # Convert grayscale to BGR for colored visualization
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if corners is not None:
        corners = corners.astype(np.int32)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(vis_image, (x, y), 3, (0, 255, 255), -1)  # Bright yellow circle
            cv2.drawMarker(vis_image, (x, y), (0, 255, 255), 
                          markerType=cv2.MARKER_CROSS, markerSize=10)  # Bright yellow cross
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image

def filter_corners_by_dot_centers(corners, dot_centers, max_distance=10):
    """
    Filter detected corners to only keep those close to known dot centers.
    
    Args:
        corners: Detected corners from Shi-Tomasi (N, 1, 2)
        dot_centers: Known dot center coordinates (M, 2)
        max_distance: Maximum distance to consider a corner as belonging to a dot
    
    Returns:
        filtered_corners: Corners that are close to dot centers
        dot_mapping: Mapping from corner index to dot index
    """
    if corners is None or len(corners) == 0:
        return None, None
    
    if len(dot_centers) == 0:
        return corners, None
    
    corners = corners.reshape(-1, 2)  # Reshape to (N, 2)
    filtered_corners = []
    dot_mapping = []
    
    for i, corner in enumerate(corners):
        # Find closest dot center
        distances = np.sqrt(np.sum((dot_centers - corner)**2, axis=1))
        closest_dot_idx = np.argmin(distances)
        min_distance = distances[closest_dot_idx]
        
        # Keep corner if it's close enough to a dot
        if min_distance <= max_distance:
            filtered_corners.append(corner)
            dot_mapping.append(closest_dot_idx)
    
    if len(filtered_corners) == 0:
        return None, None
    
    # Convert back to the format expected by Lucas-Kanade
    filtered_corners = np.array(filtered_corners).reshape(-1, 1, 2)
    dot_mapping = np.array(dot_mapping)
    
    return filtered_corners, dot_mapping

def visualize_filtered_corners(image, all_corners, filtered_corners, dot_centers, max_distance=10):
    """
    Visualize the filtering process.
    """
    # Convert grayscale to BGR for colored visualization
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw all detected corners in light gray
    if all_corners is not None:
        all_corners = all_corners.astype(np.int32)
        for corner in all_corners:
            x, y = corner.ravel()
            cv2.circle(vis_image, (x, y), 2, (128, 128, 128), -1)  # Gray
    
    # Draw dot centers in blue
    if dot_centers is not None:
        for center in dot_centers:
            x, y = int(center[0]), int(center[1])
            cv2.circle(vis_image, (x, y), 8, (255, 0, 0), 2)  # Blue circle
            cv2.circle(vis_image, (x, y), 12, (255, 0, 0), 1)  # Blue ring
    
    # Draw filtered corners in green with yellow crosses
    if filtered_corners is not None:
        filtered_corners = filtered_corners.astype(np.int32)
        for corner in filtered_corners:
            x, y = corner.ravel()
            cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)  # Green circle
            cv2.drawMarker(vis_image, (x, y), (0, 255, 255), 
                          markerType=cv2.MARKER_CROSS, markerSize=10)  # Yellow cross
    
    return vis_image

def process_corner_detection_only():
    """
    Process frames with corner detection only (no filtering).
    """
    # Create output directory
    os.makedirs('corner_detection', exist_ok=True)
    
    # Get all frames
    frame_files = sorted(glob.glob('frames_prepro/*.png'))
    
    if not frame_files:
        print("No frames found!")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # Read first frame
    first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    h, w = first_frame.shape
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('corner_detection/corner_detection.mp4', 
                           fourcc, 10.0, (w, h))
    
    # Process each frame
    corner_counts = []
    
    for i, frame_file in enumerate(frame_files):
        print(f"Processing frame {i}")
        
        # Read frame
        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        
        # Detect corners
        corners = detect_corners(frame)
        corner_counts.append(len(corners) if corners is not None else 0)
        
        # Create visualization
        vis_frame = visualize_corners(frame, corners)
        
        # Add frame information
        cv2.putText(vis_frame, f'Frame {i}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Corners: {corner_counts[-1]}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(f'corner_detection/corners_{i:03d}.png', vis_frame)
        video.write(vis_frame)
    
    video.release()
    
    print(f"\nCorner detection complete! Results saved in corner_detection/")
    print(f"- Average corners detected: {np.mean(corner_counts):.1f}")

def process_with_dot_filtering(dot_centers, max_distance=10):
    """
    Process frames with dot-based corner filtering.
    """
    # Create output directory
    os.makedirs('dot_filtered_corners', exist_ok=True)
    
    # Get all frames
    frame_files = sorted(glob.glob('frames_prepro/*.png'))
    
    if not frame_files:
        print("No frames found!")
        return
    
    print(f"Found {len(frame_files)} frames")
    print(f"Using {len(dot_centers)} dot centers")
    print(f"Maximum distance threshold: {max_distance} pixels")
    
    # Read first frame
    first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    h, w = first_frame.shape
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('dot_filtered_corners/filtered_corners.mp4', 
                           fourcc, 10.0, (w, h))
    
    # Process each frame
    all_corner_counts = []
    filtered_corner_counts = []
    
    for i, frame_file in enumerate(frame_files):
        print(f"Processing frame {i}")
        
        # Read frame
        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        
        # Detect all corners
        all_corners = detect_corners(frame)
        all_corner_counts.append(len(all_corners) if all_corners is not None else 0)
        
        # Filter corners by dot centers
        filtered_corners, dot_mapping = filter_corners_by_dot_centers(
            all_corners, dot_centers, max_distance
        )
        filtered_corner_counts.append(len(filtered_corners) if filtered_corners is not None else 0)
        
        # Create visualization
        vis_frame = visualize_filtered_corners(frame, all_corners, filtered_corners, dot_centers, max_distance)
        
        # Add frame information
        cv2.putText(vis_frame, f'Frame {i}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'All corners: {all_corner_counts[-1]}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Filtered: {filtered_corner_counts[-1]}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(f'dot_filtered_corners/filtered_{i:03d}.png', vis_frame)
        video.write(vis_frame)
    
    video.release()
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(all_corner_counts, '-', color='gray', label='All detected corners', alpha=0.7)
    plt.plot(filtered_corner_counts, '-', color='green', label='Filtered corners (near dots)', linewidth=2)
    plt.title('Corner Detection: All vs Filtered by Dot Centers')
    plt.xlabel('Frame Number')
    plt.ylabel('Corner Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('dot_filtered_corners/corner_comparison.png')
    plt.close()
    
    print(f"\nFiltering complete! Results saved in dot_filtered_corners/")
    print(f"- Average all corners: {np.mean(all_corner_counts):.1f}")
    print(f"- Average filtered corners: {np.mean(filtered_corner_counts):.1f}")
    print(f"- Filtering efficiency: {np.mean(filtered_corner_counts)/np.mean(all_corner_counts)*100:.1f}%")
