import os
import glob
import cv2
import numpy as np

def extract_frames_using_opencv(video_path, output_dir="data/frames"):
    """
    Extract all frames from a video file using OpenCV.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory where to save the frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to 8-bit, scaling up by 16 to preserve full range
        frame = (frame * 16).clip(0, 255).astype(np.uint8)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame {frame_count}")
        frame_count += 1
    
    cap.release()
    print(f"\nExtracted {frame_count-1} frames to {output_dir}")

def process_video_folder(folder_path="data/cine_videos", frames_dir="data/frames"):
    """
    Process all video files in a folder and extract frames using OpenCV.
    
    Args:
        folder_path (str): Path to the folder containing video files
        frames_dir (str): Directory where to save extracted frames
    """
    # Create folders if they don't exist
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get CamA video file
    video_path = os.path.join(folder_path, "CamA_DPZM_04_cc_lo_01_output.avi")
    if not os.path.exists(video_path):
        print(f"CamA video not found at {video_path}")
        return
    
    # Process video file
    print(f"\nProcessing {os.path.basename(video_path)}...")
    extract_frames_using_opencv(video_path, frames_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video files using OpenCV")
    parser.add_argument("--video-dir", default="data/cine_videos", help="Directory containing video files")
    parser.add_argument("--frames-dir", default="data/frames", help="Directory to save extracted frames")
    
    args = parser.parse_args()
    
    print("\nStarting frame extraction...")
    print(f"Input directory: {args.video_dir}")
    print(f"Output frames directory: {args.frames_dir}\n")
    
    process_video_folder(
        folder_path=args.video_dir,
        frames_dir=args.frames_dir
    )

