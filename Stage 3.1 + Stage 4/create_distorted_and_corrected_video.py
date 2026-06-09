import cv2
import os
import glob
import numpy as np

def count_frames_in_directory(frames_dir):
    """
    Count the number of PNG frames in a directory.
    
    Args:
        frames_dir (str): Directory containing the frames
        
    Returns:
        int: Number of PNG frames found
    """
    frame_pattern = os.path.join(frames_dir, "*.png")
    frame_files = glob.glob(frame_pattern)
    return len(frame_files)

def create_video_from_frames(frames_dir, output_path, fps=30, max_frames=None):
    """
    Create a video from frames with specified frame rate.
    
    Args:
        frames_dir (str): Directory containing the frames
        output_path (str): Path for the output video file
        fps (int): Frames per second for the output video
        max_frames (int): Maximum number of frames to use (None for all)
    """
    
    # Get all frame files and sort them numerically
    frame_pattern = os.path.join(frames_dir, "*.png")
    frame_files = glob.glob(frame_pattern)
    
    if not frame_files:
        print(f"No PNG frames found in {frames_dir}")
        return False
    
    # Sort frames by frame number (handle different naming patterns)
    def extract_frame_number(filename):
        basename = os.path.basename(filename)
        # Try to extract number from different patterns
        if 'corrected_frame_' in basename:
            return int(basename.split('corrected_frame_')[1].split('.')[0])
        elif 'frame_' in basename:
            return int(basename.split('frame_')[1].split('.')[0])
        elif basename.replace('.png', '').isdigit():
            return int(basename.replace('.png', ''))
        else:
            # Fallback to alphabetical sorting
            return basename
    
    frame_files.sort(key=extract_frame_number)
    
    # Limit frames if max_frames is specified
    if max_frames is not None and len(frame_files) > max_frames:
        frame_files = frame_files[:max_frames]
        print(f"Limited to first {max_frames} frames from {frames_dir}")
    
    print(f"Using {len(frame_files)} frames from {frames_dir}")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Could not read first frame: {frame_files[0]}")
        return False
    
    height, width, layers = first_frame.shape
    print(f"Frame dimensions: {width}x{height}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return False
    
    # Write each frame to the video
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        
        video_writer.write(frame)
        
        # Print progress every 10 frames
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames")
    
    # Release everything
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video created successfully: {output_path}")
    print(f"Video specs: {width}x{height}, {fps} fps, {len(frame_files)} frames")
    print(f"Duration: {len(frame_files)/fps:.2f} seconds")
    return True

def get_user_choice():
    """
    Get user choice for which frames to use for video creation.
    
    Returns:
        tuple: (frames_directory, output_filename_prefix, max_frames)
    """
    print("\n" + "="*50)
    print("VIDEO CREATION TOOL")
    print("="*50)
    print("Choose which frames to use for video creation:")
    print("1) Distorted video (using 'data/frames' directory)")
    print("2) Corrected video (using 'data/corrected_frames' directory)")
    print("="*50)
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                frames_dir = "data/frames"
                output_prefix = "distorted_video"
                
                # Count frames in corrected_frames directory
                corrected_frames_count = count_frames_in_directory("data/corrected_frames")
                print(f"Found {corrected_frames_count} frames in corrected_frames directory")
                print(f"Will limit distorted video to {corrected_frames_count} frames for comparison")
                print(f"Selected: Distorted video from '{frames_dir}' directory (limited to {corrected_frames_count} frames)")
                break
            elif choice == '2':
                frames_dir = "data/corrected_frames"
                output_prefix = "corrected_video"
                print(f"Selected: Corrected video from '{frames_dir}' directory")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None, None, None
    
    return frames_dir, output_prefix, corrected_frames_count if choice == '1' else None

def main():
    """
    Main function to create video from user-selected frames.
    """
    # Get user choices
    frames_dir, output_prefix, max_frames = get_user_choice()
    if frames_dir is None:
        return
    
    # Always use 30 fps
    fps = 30
    
    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        print(f"Error: Directory '{frames_dir}' does not exist!")
        return
    
    # Create output filename
    output_filename = f"data/{output_prefix}_30fps.mp4"
    
    print(f"\nCreating video:")
    print(f"  Source: {frames_dir}/")
    print(f"  Output: {output_filename}")
    print(f"  Frame rate: {fps} fps (fixed)")
    if max_frames:
        print(f"  Max frames: {max_frames}")
    print("-" * 50)
    
    # Create the video
    success = create_video_from_frames(frames_dir, output_filename, fps, max_frames)
    
    if success:
        print(f"\nVideo creation completed successfully!")
        print(f"Output file: {output_filename}")
    else:
        print(f"\nVideo creation failed!")

if __name__ == "__main__":
    main()
