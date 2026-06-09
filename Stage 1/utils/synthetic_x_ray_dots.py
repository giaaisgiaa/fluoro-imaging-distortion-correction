import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

def is_valid_position(x, y, dot_locations, min_distance):
    """Check if a new dot position is far enough from existing dots"""
    for dot in dot_locations:
        dx = dot["x"] - x
        dy = dot["y"] - y
        distance = np.sqrt(dx*dx + dy*dy)
        if distance < min_distance:
            return False
    return True

def is_image_too_dark(img_array, threshold=10):
    """Check if image is too dark (mostly black)"""
    mean_intensity = np.mean(img_array)
    return mean_intensity < threshold

def generate_xray_dataset(num_images=200, image_size=256, min_dots=50, max_dots=150, 
                         dot_radius=10, output_dir="dataset", max_attempts=1000):
    """
    Generate a dataset of images with dots matching real X-ray characteristics.
    All processing is done in 0-1 range (like the model sees it) and converted to 0-255 only for saving.
    
    Background: ~0.35-0.41 (90-104 in 0-255 scale)
    Dots: ~0.98-0.99 (250-253 in 0-255 scale)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Minimum distance between dot centers
    min_distance = 2.5 * dot_radius  # A bit more than 2*radius for some padding
    
    images_generated = 0
    attempts = 0
    max_image_attempts = 50  # Maximum attempts to generate a valid image
    
    # X-ray characteristics in 0-1 range
    bg_color = 95/255  # ~0.37
    bg_min = 90/255    # ~0.35
    bg_max = 104/255   # ~0.41
    
    dot_intensities = [
        # Very bright dots (70% chance)
        {
            "mean": 252/255,  # ~0.988
            "std": 1/255,     # ~0.004
            "min": 250/255,   # ~0.980
            "max": 253/255,   # ~0.992
            "weight": 0.7,
            "blur_range": (0.8, 1.2)  # Less blur for sharp dots
        },
        # Blurred, lower intensity dots (30% chance)
        {
            "mean": 180/255,  # ~0.706
            "std": 15/255,    # More variation
            "min": 150/255,   # ~0.588
            "max": 210/255,   # ~0.824
            "weight": 0.3,
            "blur_range": (1.5, 2.5)  # More blur for distorted dots
        }
    ]
    
    while images_generated < num_images and attempts < max_image_attempts:
        # Create background with realistic variation in 0-1 range
        bg_variation = np.random.normal(0, 3/255, (image_size, image_size))
        background = np.clip(bg_color + bg_variation, bg_min, bg_max)
        
        # Convert to uint8 for PIL
        background_uint8 = (background * 255).astype(np.uint8)
        img = Image.fromarray(background_uint8)
        draw = ImageDraw.Draw(img)
        
        # Determine number of dots for this image
        num_dots = random.randint(min_dots, max_dots)
        
        # Store dot locations
        dot_locations = []
        
        # Generate random non-overlapping dots
        dot_attempts = 0
        dots_placed = 0
        
        while dots_placed < num_dots and dot_attempts < max_attempts:
            # Ensure dots are fully within the image
            x = random.randint(dot_radius, image_size - dot_radius)
            y = random.randint(dot_radius, image_size - dot_radius)
            
            if is_valid_position(x, y, dot_locations, min_distance):
                # Choose dot type based on weights
                rand = random.random()
                cumulative_weight = 0
                for dot_type in dot_intensities:
                    cumulative_weight += dot_type["weight"]
                    if rand <= cumulative_weight:
                        break
                
                # Generate dot intensity in 0-1 range
                dot_intensity_norm = np.clip(
                    np.random.normal(dot_type["mean"], dot_type["std"]),
                    dot_type["min"],
                    dot_type["max"]
                )
                
                # Convert to uint8 for drawing
                dot_intensity = int(dot_intensity_norm * 255)
                
                # Create gradient effect for the dot
                dot_size = 2 * dot_radius + 1
                dot_img = Image.new('L', (dot_size, dot_size), int(background_uint8[y, x]))
                dot_draw = ImageDraw.Draw(dot_img)
                
                # Draw multiple circles with decreasing intensity for gradient effect
                num_gradient_steps = 5
                for i in range(num_gradient_steps):
                    current_radius = dot_radius * (1 - i/num_gradient_steps)
                    # Calculate gradient in normalized space
                    current_intensity_norm = dot_intensity_norm - (dot_intensity_norm - bg_color) * (i/num_gradient_steps)
                    # Convert to uint8 for drawing
                    current_intensity = int(current_intensity_norm * 255)
                    dot_draw.ellipse(
                        [(dot_radius - current_radius, dot_radius - current_radius),
                         (dot_radius + current_radius, dot_radius + current_radius)],
                        fill=current_intensity
                    )
                
                # Apply gaussian blur with type-specific range
                min_blur, max_blur = dot_type["blur_range"]
                blur_radius = random.uniform(min_blur, max_blur)
                dot_img = dot_img.filter(ImageFilter.GaussianBlur(blur_radius))
                
                # Paste the dot
                img.paste(dot_img, (x - dot_radius, y - dot_radius))
                
                # Store location and both normalized and uint8 intensities
                dot_locations.append({
                    "x": int(x),
                    "y": int(y),
                    "intensity_norm": float(dot_intensity_norm),
                    "intensity_uint8": int(dot_intensity),
                    "is_bright": dot_type["mean"] > 200/255,  # Flag if it's a bright dot
                    "blur_amount": float(blur_radius)
                })
                dots_placed += 1
                dot_attempts = 0  # Reset attempts for next dot
            else:
                dot_attempts += 1
        
        # Convert to numpy array to check final image
        final_img_array = np.array(img)
        if is_image_too_dark(final_img_array):
            attempts += 1
            continue
        
        # If we couldn't place all dots, print a warning
        if dots_placed < num_dots:
            print(f"Warning: Could only place {dots_placed} dots out of {num_dots} requested in image {images_generated+1}")
        
        # Save the image (already in uint8 format)
        img_path = os.path.join(output_dir, f"xray_image_{images_generated:04d}.png")
        img.save(img_path)
        
        # Save the JSON file with dot locations
        json_path = os.path.join(output_dir, f"xray_image_{images_generated:04d}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "image_size": int(image_size),
                "dot_radius": int(dot_radius),
                "background_color_norm": float(bg_color),
                "background_color_uint8": int(bg_color * 255),
                "dots": dot_locations,
                "dots_placed": int(dots_placed),
                "dots_requested": int(num_dots)
            }, f, indent=2)
        
        print(f"Generated X-ray image {images_generated+1}/{num_images} with {dots_placed} dots")
        images_generated += 1
        attempts = 0  # Reset attempts counter after successful generation
    
    if images_generated < num_images:
        print(f"\nWarning: Could only generate {images_generated} valid images out of {num_images} requested") 