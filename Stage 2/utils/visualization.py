import cv2
import numpy as np
import os

def overlap_distorted_coordinates(num_dots, num_frames, output_video="distorted_coordinates_overlay.mp4"):
    """
    Create a video showing original frames with distorted trajectory coordinates overlaid in red.
    
    Args:
        num_dots: Number of dots/trajectories
        num_frames: Number of frames to process
        output_video: Output video filename
    """
    # Get frame dimensions from first frame
    first_frame_path = os.path.join("data/frames", "frame_1.png")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 10.0, (width, height))  # 10 FPS
    
    print(f"Processing {num_frames} frames with {num_dots} dots each...")
    
    for frame_num in range(1, num_frames + 1):
        # Load original frame
        frame_path = os.path.join("data/frames", f"frame_{frame_num}.png")
        if not os.path.exists(frame_path):
            print(f"Warning: Frame {frame_num} not found, skipping...")
            continue
            
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not load frame {frame_num}, skipping...")
            continue
        
        # Load all trajectory coordinates for this frame
        all_coordinates = []
        for dot_index in range(num_dots):
            trajectory_file = os.path.join("data/trajectories", f"trajectory_{dot_index}.txt")
            if os.path.exists(trajectory_file):
                try:
                    trajectory = np.loadtxt(trajectory_file, skiprows=1)  # Skip header
                    if frame_num <= len(trajectory):
                        # Get the coordinate for this frame (frame_num-1 because arrays are 0-indexed)
                        coord = trajectory[frame_num - 1]
                        if not np.isnan(coord[0]) and not np.isnan(coord[1]):
                            all_coordinates.append((int(coord[0]), int(coord[1])))
                except Exception as e:
                    print(f"Warning: Error loading trajectory {dot_index}: {e}")
                    continue
        
        # Draw red dots for all coordinates
        for coord in all_coordinates:
            x, y = coord
            # Draw a red circle at each coordinate
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red circle with radius 3
        
        # Add frame number text
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Dots: {len(all_coordinates)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        if frame_num % 10 == 0:
            print(f"Processed frame {frame_num}/{num_frames}")
    
    # Release video writer
    out.release()
    print(f"Video saved as: {output_video}")
    
    return output_video



