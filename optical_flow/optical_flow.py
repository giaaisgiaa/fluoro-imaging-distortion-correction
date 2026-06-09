import cv2
import numpy as np
import os
import glob

def extract_motion_vectors(frame1, frame2):
    """
    Extract motion vectors between two consecutive frames using optical flow.
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Parameters for Lucas-Kanade optical flows
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Detect good features to track
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=100,
                                qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None and len(p0) > 0:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
        
        # Select good points
        good_old = p0[st==1]
        good_new = p1[st==1]
        
        # Create motion vectors
        motion_vectors = []
        for old, new in zip(good_old, good_new):
            start_pt = (int(old[0]), int(old[1]))
            end_pt = (int(new[0]), int(new[1]))
            motion_vectors.append((start_pt, end_pt))
        
        return motion_vectors
    else:
        return []

def visualize_motion_vectors(frame, motion_vectors):
    """
    Visualize motion vectors as green arrows on the frame.
    """
    vis_frame = frame.copy()
    
    # Draw green arrows for each motion vector
    for start_pt, end_pt in motion_vectors:
        # Calculate motion magnitude
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Only draw arrows for significant motion (more than 1 pixel)
        if magnitude > 1:
            # Draw green arrow
            cv2.arrowedLine(vis_frame, start_pt, end_pt, (0, 255, 0), 2, tipLength=0.3)
            
            # Draw small circle at start point
            cv2.circle(vis_frame, start_pt, 3, (0, 255, 0), -1)
    
    return vis_frame

def create_motion_video(image_files):
    """
    Create a video showing motion vectors across all frames.
    """
    if len(image_files) < 2:
        print("Need at least 2 images for motion visualization")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(image_files[0])
    h, w = first_frame.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('optical_flow_motion.mp4', fourcc, 30.0, (w, h))
    
    # Process each pair of consecutive frames
    for i in range(len(image_files) - 1):
        print(f"Processing frames {i} → {i+1}")
        
        # Read consecutive frames
        frame1 = cv2.imread(image_files[i])
        frame2 = cv2.imread(image_files[i + 1])
        
        # Extract motion vectors
        motion_vectors = extract_motion_vectors(frame1, frame2)
        
        # Visualize motion vectors on frame2
        vis_frame = visualize_motion_vectors(frame2, motion_vectors)
        
        # Add frame information
        cv2.putText(vis_frame, f'Frame {i+1}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Motion vectors: {len(motion_vectors)}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        video_writer.write(vis_frame)
    
    # Release video writer
    video_writer.release()
    
    print(f"Motion video saved as 'optical_flow_motion.mp4'")

def main():
    # Get all image files
    image_files = sorted(glob.glob('frames/*.png'))
    
    if not image_files:
        print("No image files found")
        return
    
    print(f"Found {len(image_files)} images")
    print("Creating motion visualization...")
    
    # Create the motion video
    create_motion_video(image_files)
    
    print("Done! Motion video created with green motion vectors.")

if __name__ == "__main__":
    main() 