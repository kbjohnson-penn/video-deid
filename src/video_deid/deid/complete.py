"""
Complete video de-identification functionality
"""
import cv2
import logging
from pathlib import Path

from ..utils.progress import create_progress_bar
from ..config import (
    BLUR_KERNEL_SIZE, BLUR_SIGMA, DEFAULT_FOURCC,
    LOG_FREQUENCY, BATCH_SIZE, SKELETON_CONNECTIONS,
    KEYPOINT_RADIUS, KEYPOINT_COLOR, LINE_THICKNESS
)


def blur_entire_frame(frame, kernel_size=None, sigma=None):
    """
    Apply a Gaussian blur to the entire frame for privacy protection.

    Parameters:
    - frame: The input video frame to blur
    - kernel_size: Size of the Gaussian blur kernel (default from config)
    - sigma: Sigma value for Gaussian blur (default from config)

    Returns:
    - blurred_frame: The completely blurred frame
    """
    if kernel_size is None:
        # Use odd-sized kernel for better results
        kernel_size = (BLUR_KERNEL_SIZE*2+1, BLUR_KERNEL_SIZE*2+1)

    if sigma is None:
        sigma = BLUR_SIGMA

    blurred_frame = cv2.GaussianBlur(frame, kernel_size, sigma)
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
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_FOURCC)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        out = cv2.VideoWriter(temp_blurred_video_path_str,
                              fourcc, fps, (width, height))

        # Set up progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = create_progress_bar(
            total_frames, "Blurring video", show_progress)

        # Process frames in batches for better performance
        frame_count = 0

        # Process in batches for better performance
        while frame_count < total_frames:
            # Read a batch of frames
            frames_batch = []
            batch_size = min(BATCH_SIZE, total_frames - frame_count)

            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)

            if not frames_batch:
                break

            # Process the batch (could be parallelized if needed)
            blurred_frames = [blur_entire_frame(
                frame) for frame in frames_batch]

            # Write the processed batch
            for blurred_frame in blurred_frames:
                out.write(blurred_frame)

                if progress_bar:
                    progress_bar.update(1)

            # Update frame count
            batch_frame_count = len(frames_batch)
            frame_count += batch_frame_count

            # Log periodically to avoid console spam
            if frame_count % LOG_FREQUENCY == 0 or frame_count == total_frames:
                logging.debug(
                    f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames:.1%})")

        logging.info(
            f"Blurred {frame_count} frames, saved to: {temp_blurred_video_path}")

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
    # Convert coordinates to integers once
    int_keypoints = []
    valid_keypoints = []

    for i, (x, y) in enumerate(keypoints):
        if x == 0.0 and y == 0.0:  # Invalid keypoint
            int_keypoints.append(None)
            valid_keypoints.append(False)
        else:
            int_keypoints.append((int(x), int(y)))
            valid_keypoints.append(True)
            # Draw keypoint
            cv2.circle(frame, int_keypoints[-1],
                       KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)

    # Draw skeleton connections
    for (start_idx, end_idx), color in SKELETON_CONNECTIONS:
        # Only draw if both points are valid
        if valid_keypoints[start_idx] and valid_keypoints[end_idx]:
            cv2.line(frame, int_keypoints[start_idx], int_keypoints[end_idx],
                     color, LINE_THICKNESS)

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
            logging.error(
                f"Failed to open video source: {blurred_video_source}")
            raise IOError(f"Cannot open video: {blurred_video_source}")

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_FOURCC)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        out = cv2.VideoWriter(output_video_path_str,
                              fourcc, fps, (width, height))

        # Set up progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = create_progress_bar(
            total_frames, "Applying keypoints to blurred video", show_progress)

        # Determine if frame numbers start at 0 or 1
        # Check if we have frame_number 0 in the dataframe
        has_zero_index = 0 in df['frame_number'].values
        start_frame = 0 if has_zero_index else 1

        # Process frames in batches for better performance
        frames_processed = 0

        while frames_processed < total_frames:
            # Read a batch of frames
            frames_batch = []
            frame_numbers = []
            batch_size = min(BATCH_SIZE, total_frames - frames_processed)

            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate the actual frame number in the dataframe
                frame_number = frames_processed + start_frame

                frames_batch.append(frame)
                frame_numbers.append(frame_number)
                frames_processed += 1

            if not frames_batch:
                break

            # Process each frame in the batch
            for i, (frame, frame_number) in enumerate(zip(frames_batch, frame_numbers)):
                # Get only the keypoints for current frame
                frame_df = df[df['frame_number'] == frame_number]

                if not frame_df.empty:
                    frame_copy = frame.copy()

                    # Process each person's keypoints in this frame
                    for _, row in frame_df.iterrows():
                        keypoints = [(row[f'x_{i}'], row[f'y_{i}'])
                                     for i in range(0, 17)]
                        frame_copy = draw_annotations(frame_copy, keypoints)

                    out.write(frame_copy)
                else:
                    # No keypoints for this frame, just write original frame
                    out.write(frame)

                if progress_bar:
                    progress_bar.update(1)

            # Log progress periodically
            if frames_processed % LOG_FREQUENCY == 0 or frames_processed == total_frames:
                logging.debug(
                    f"Processed {frames_processed}/{total_frames} frames with keypoints ({frames_processed/total_frames:.1%})")

        logging.info(
            f"Processed {frames_processed} frames with keypoints, saved to: {output_video_path}")

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
