"""
Core functionality for video de-identification
"""
import logging
import cv2

from helpers.utils import get_video_properties
from .tracking import generate_kalman_predictions
from .processing import initialize_video_writer, process_frame_loop


def process_video(video_path, keypoints_df, interpolated_keypoints_df, kalman_filtered_keypoints_df_path, output_path, show_progress):
    """
    Processes a video based on the kalman filter predictions and saves the output to a file.

    Parameters:
    - video_path (str): Path to the input video.
    - keypoints_df (pd.DataFrame): Dataframe containing keypoints.
    - interpolated_keypoints_df (pd.DataFrame): Dataframe containing interpolated keypoints.
    - kalman_filtered_keypoints_df_path (str): Path to the Kalman filtered keypoints dataframe.
    - output_path (str): Path to the output video.
    - show_progress (bool): Show progress bar.

    Returns:
    - None
    """
    cap = None
    out = None
    
    try:
        # Step 1: Get video properties
        frame_width, frame_height, fps, total_frames = get_video_properties(video_path)
        logging.info('Getting video properties.')

        # Step 2: Generate Kalman predictions
        logging.info('Generating Kalman predictions')
        predicted_df = generate_kalman_predictions(
            keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames)
        predicted_df.to_csv(kalman_filtered_keypoints_df_path, index=False)

        # Step 3: Initialize video capture and writer
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        out = initialize_video_writer(video_path, output_path, fps, frame_width, frame_height)
        logging.info('Starting video processing...')

        # Step 4: Process frames
        process_frame_loop(cap, predicted_df, out, show_progress, total_frames)
        logging.info('Video processing completed successfully.')

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except IOError as e:
        logging.error(f"IO error occurred: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred in process_video: {e}")

    finally:
        # Step 5: Clean up resources
        if cap:
            cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()