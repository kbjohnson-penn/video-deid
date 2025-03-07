"""
File and I/O utility functions
"""
import os
import logging
import cv2
import time
from pathlib import Path
from ..config import DEFAULT_RUNS_DIR


def create_run_directory_and_paths(video_path):
    """
    Creates a run directory based on the video file name and current timestamp,
    and returns the paths for the run directory, log file, interpolated CSV, and Kalman filtered CSV.

    Parameters:
    - video_path (str or Path): Path to the input video.

    Returns:
    - dict: Paths for the run directory, log file, interpolated CSV, and Kalman filtered CSV.
    """
    video_path = Path(video_path)
    video_file_name = video_path.stem
    time_stamp = int(time.time())

    # Ensure the runs directory exists
    DEFAULT_RUNS_DIR.mkdir(exist_ok=True)

    current_run = DEFAULT_RUNS_DIR / f"{video_file_name}_{time_stamp}"
    current_run.mkdir(exist_ok=True)
    logging.info(f"Created current run directory: {current_run}")

    paths = {
        'run_directory': str(current_run),
        'log_file': str(current_run / f"{video_file_name}_{time_stamp}.log"),
        'interpolated_csv': str(current_run / f"{video_file_name}_interpolated.csv"),
        'kalman_filtered_csv': str(current_run / f"{video_file_name}_kalman_filtered.csv")
    }
    return paths


def setup_logging(log_file=None):
    """
    Set up logging to ensure that existing handlers are removed, and new handlers are configured correctly.

    Parameters:
    - log_file (str, optional): Path to the log file.

    Returns:
    - None
    """
    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to add file handler to logger: {e}")

    logging.info("Logging has been configured.")


def get_video_properties(video_path):
    """
    Extracts the dimensions of the video frames, frame rate and total number of frames.

    Parameters:
    - video_path (str or Path): Path to the video file.

    Returns:
    - tuple: The frame width, frame height, frame rate, and total number of frames.
    """
    video_path = str(video_path)  # Convert Path objects to string for OpenCV
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logging.error(
                f"Cannot open video: {video_path}. Please check if the path is correct or if the file is corrupted.")
            raise IOError(f"Cannot open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    logging.info(
        f"Video properties: width={frame_width}, height={frame_height}, fps={fps}, total_frames={total_frames}")
    return frame_width, frame_height, fps, total_frames
