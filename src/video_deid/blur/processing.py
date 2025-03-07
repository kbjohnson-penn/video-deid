"""
Frame processing functions for video de-identification
"""
import logging
import cv2
import pandas as pd
import numpy as np

from ..utils import calculate_bounding_box, create_progress_bar
from .techniques import apply_circular_blur, apply_full_frame_blur
from ..config import DEFAULT_FOURCC, LOG_FREQUENCY, FACE_MARGIN


def process_frame(frame, estimated_x, estimated_y, blur_faces=False, draw_bbox=True):
    """
    Processes a frame and applies blur to faces.

    Parameters:
    - frame (np.array): The frame to process.
    - estimated_x (int): The estimated x-coordinate of the face.
    - estimated_y (int): The estimated y-coordinate of the face.
    - blur_faces (bool): Whether to blur faces.
    - draw_bbox (bool): Whether to draw bounding boxes.

    Returns:
    np.array: The processed frame.
    """
    # Validate keypoints
    if estimated_x is None or estimated_y is None:
        logging.warning(f"Invalid keypoints, applying full-frame blur")
        return apply_full_frame_blur(frame)

    # Check if keypoints are within reasonable bounds (outside frame or at extreme edges)
    height, width = frame.shape[:2]
    border_margin = 10

    if (estimated_x < border_margin or estimated_x > width - border_margin or
            estimated_y < border_margin or estimated_y > height - border_margin):
        logging.warning(
            f"Keypoints out of bounds: ({estimated_x}, {estimated_y}), applying full-frame blur")
        return apply_full_frame_blur(frame)

    if draw_bbox or blur_faces:
        try:
            x_min, y_min, x_max, y_max = calculate_bounding_box(
                np.array([[estimated_x, estimated_y]]), frame.shape, margin=FACE_MARGIN)

            # Validate that we have a reasonable bounding box
            if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                logging.warning(
                    f"Bounding box too small: width={x_max-x_min}, height={y_max-y_min}, applying full-frame blur")
                return apply_full_frame_blur(frame)

            if blur_faces:
                frame = apply_circular_blur(frame, x_min, y_min, x_max, y_max)
            if draw_bbox:
                from .techniques import draw_bounding_box_and_keypoints
                frame = draw_bounding_box_and_keypoints(
                    frame, [[estimated_x, estimated_y]], x_min, y_min, x_max, y_max)
        except Exception as e:
            logging.error(f"Error in process_frame: {e}")
            return apply_full_frame_blur(frame)

    return frame


def initialize_video_writer(video_path, output_path, fps, frame_width=None, frame_height=None):
    """
    Initializes a video writer object.

    Parameters:
    - video_path (str): Path to the input video.
    - output_path (str): Path to the output video.
    - fps (float): Frame rate of the video.
    - frame_width (int, optional): Width of the video frame.
    - frame_height (int, optional): Height of the video frame.

    Returns:
    - cv2.VideoWriter: Video writer object.
    """
    if frame_width is None or frame_height is None:
        from ..utils import get_video_properties
        frame_width, frame_height, _, _ = get_video_properties(video_path)

    fourcc = cv2.VideoWriter_fourcc(*DEFAULT_FOURCC)
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


def process_frame_loop(cap, predicted_df, out, show_progress, total_frames):
    """
    Processes frames in a video and applies Gaussian blur to faces.
    If no valid predictions are found, applies a light blur to the entire frame.

    Parameters:
    - cap (cv2.VideoCapture): Video capture object.
    - predicted_df (pd.DataFrame): Dataframe containing Kalman filter predictions.
    - out (cv2.VideoWriter): Video writer object.
    - show_progress (bool): Show progress bar.
    - total_frames (int): Total number of frames in the video.

    Returns:
    - None
    """
    # Pre-process predictions into a dictionary for faster lookup
    predictions_by_frame = {}
    for frame_num, frame_group in predicted_df.groupby('frame_number'):
        valid_predictions = []
        for _, row in frame_group.iterrows():
            if pd.notna(row['estimated_x']) and pd.notna(row['estimated_y']):
                valid_predictions.append(
                    (row['person_id'], row['estimated_x'], row['estimated_y']))
        if valid_predictions:
            predictions_by_frame[frame_num] = valid_predictions

    frame_number = 1
    progress_bar = create_progress_bar(
        total_frames, "Processing and blurring video", show_progress)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get predictions for current frame from dictionary (fast lookup)
        predictions = predictions_by_frame.get(frame_number, [])

        # Always create a copy to be safe
        frame_copy = frame.copy()

        if predictions:
            # Process all faces in the frame
            for person_id, est_x, est_y in predictions:
                try:
                    # Reduce logging to improve performance
                    if frame_number % LOG_FREQUENCY == 0:
                        logging.info(
                            f"Processing frame {frame_number} for person_id {person_id}")

                    # Apply blur directly
                    frame_copy = process_frame(
                        frame_copy, est_x, est_y, blur_faces=True, draw_bbox=False
                    )
                except Exception as e:
                    logging.error(
                        f"Error processing frame {frame_number}: {e}")
                    # Make sure this frame still gets blurred
                    frame_copy = apply_full_frame_blur(frame_copy)
        else:
            # No valid keypoints found for this frame, apply blur to entire frame
            if frame_number % LOG_FREQUENCY == 0:
                logging.info(
                    f"No valid keypoints for frame {frame_number}, applying full-frame blur")
            frame_copy = apply_full_frame_blur(frame_copy)

        # Make sure we always write a frame
        out.write(frame_copy)

        frame_number += 1
        if show_progress:
            progress_bar.update(1)

    if show_progress:
        progress_bar.close()
