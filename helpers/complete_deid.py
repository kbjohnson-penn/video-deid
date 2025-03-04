import cv2
import logging
import shutil
from pathlib import Path
from helpers.utils import create_progress_bar


def blur_entire_frame(frame):
    """
    Apply a Gaussian blur to the entire frame for privacy protection.
    
    Parameters:
    - frame: The input video frame to blur
    
    Returns:
    - blurred_frame: The completely blurred frame
    """
    kernel_size = (151, 151)
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)
    return blurred_frame


def blur_video(video_source, temp_blurred_video_path, show_progress=False):
    """
    Process a video by applying Gaussian blur to every frame for privacy protection.
    
    Parameters:
    - video_source: Path to the input video
    - temp_blurred_video_path: Path to save the blurred video
    - show_progress: Whether to display a progress bar
    """
    # Convert to Path objects and then to strings for OpenCV
    video_source = Path(video_source)
    temp_blurred_video_path = Path(temp_blurred_video_path)
    
    # Ensure output directory exists
    temp_blurred_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to strings for OpenCV
    video_source_str = str(video_source)
    temp_blurred_video_path_str = str(temp_blurred_video_path)
    
    cap = None
    out = None
    progress_bar = None
    
    try:
        cap = cv2.VideoCapture(video_source_str)
        if not cap.isOpened():
            logging.error(f"Failed to open video source: {video_source}")
            raise IOError(f"Cannot open video: {video_source}")

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        out = cv2.VideoWriter(temp_blurred_video_path_str, fourcc, fps, (width, height))

        # Set up progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = create_progress_bar(total_frames, "Blurring video", show_progress)

        # Process each frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            blurred_frame = blur_entire_frame(frame)
            out.write(blurred_frame)
            
            if progress_bar:
                progress_bar.update(1)
                
            frame_count += 1

        logging.info(f"Blurred {frame_count} frames, saved to: {temp_blurred_video_path}")

    except Exception as e:
        logging.error(f"Failed to process video {video_source}: {e}")
        raise
    finally:
        # Ensure resources are always released
        if cap and cap.isOpened():
            cap.release()
        if out:
            out.release()
        if progress_bar:
            progress_bar.close()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.debug(f"Could not destroy OpenCV windows: {e}")


def draw_annotations(frame, keypoints):
    """
    Draw skeleton annotations on a frame using keypoints.
    
    Parameters:
    - frame: The video frame to draw on
    - keypoints: List of (x,y) coordinates for body keypoints
    
    Returns:
    - The frame with skeleton annotations
    """
    # Define skeleton connections with appropriate colors
    skeleton = [
        ((0, 1), 'green'), ((0, 2), 'green'), ((1, 3), 'green'), ((2, 4), 'green'),
        ((3, 5), 'blue'), ((4, 6), 'blue'), ((5, 6), 'blue'),
        ((5, 7), 'blue'), ((7, 9), 'blue'), ((6, 8), 'blue'), ((8, 10), 'blue'),
        ((5, 11), 'orange'), ((6, 12), 'orange'), ((11, 12), 'orange'),
        ((11, 13), 'orange'), ((13, 15), 'orange'), ((12, 14), 'orange'), ((14, 16), 'orange')
    ]
    
    color_map = {
        'green': (0, 255, 0), 
        'blue': (0, 0, 255), 
        'orange': (0, 165, 255)
    }

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if not (x == 0.0 and y == 0.0):  # Skip invalid keypoints
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Draw skeleton connections
    for (start_point, end_point), color in skeleton:
        start_x, start_y = keypoints[start_point]
        end_x, end_y = keypoints[end_point]
        
        # Only draw if both points are valid
        if not (start_x == 0.0 and start_y == 0.0) and not (end_x == 0.0 and end_y == 0.0):
            cv2.line(frame, (int(start_x), int(start_y)),
                    (int(end_x), int(end_y)), color_map[color], 2)

    return frame


def process_blurred_video(blurred_video_source, df, output_video_path, show_progress=False):
    """
    Process a blurred video by overlaying keypoint skeletons.
    
    Parameters:
    - blurred_video_source: Path to the blurred video
    - df: DataFrame containing keypoints data
    - output_video_path: Path to save the output video with keypoints
    - show_progress: Whether to display a progress bar
    """
    # Convert to Path objects and then to strings for OpenCV
    blurred_video_source = Path(blurred_video_source)
    output_video_path = Path(output_video_path)
    
    # Ensure output directory exists
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to strings for OpenCV
    blurred_video_source_str = str(blurred_video_source)
    output_video_path_str = str(output_video_path)
    
    cap = None
    out = None
    progress_bar = None
    
    try:
        cap = cv2.VideoCapture(blurred_video_source_str)
        if not cap.isOpened():
            logging.error(f"Failed to open video source: {blurred_video_source}")
            raise IOError(f"Cannot open video: {blurred_video_source}")

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        out = cv2.VideoWriter(output_video_path_str, fourcc, fps, (width, height))

        # Set up progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = create_progress_bar(
            total_frames, "Applying keypoints to blurred video", show_progress)

        # Process each frame
        for frame_number in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Get only the keypoints for current frame
            frame_df = df[df['frame_number'] == frame_number]
            
            if not frame_df.empty:
                frame_copy = frame.copy()
                
                # Process each person's keypoints in this frame
                for _, row in frame_df.iterrows():
                    keypoints = [(row[f'x_{i}'], row[f'y_{i}']) for i in range(0, 17)]
                    frame_copy = draw_annotations(frame_copy, keypoints)
                    
                out.write(frame_copy)
            else:
                # No keypoints for this frame, just write original frame
                out.write(frame)
                
            if progress_bar:
                progress_bar.update(1)

        logging.info(f"Processed video with keypoints saved to: {output_video_path}")

    except Exception as e:
        logging.error(f"Failed to process video: {e}")
        raise
    finally:
        # Ensure resources are always released
        if cap and cap.isOpened():
            cap.release()
        if out:
            out.release()
        if progress_bar:
            progress_bar.close()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.debug(f"Could not destroy OpenCV windows: {e}")
