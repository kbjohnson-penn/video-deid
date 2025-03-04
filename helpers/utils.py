import os
import logging
import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm


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
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    current_run = runs_dir / f"{video_file_name}_{time_stamp}"
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


def calculate_time(frame_number, frame_rate):
    """ Calculates the time in the video given the frame number and the frame rate. """
    return frame_number / frame_rate


def scale_keypoints(keypoints, width, height):
    """
    Scales keypoints to the frame dimensions.

    Parameters:
    - keypoints (list): The keypoints to scale.
    - width (int): The width of the frame.
    - height (int): The height of the frame.

    Returns:
    - np.ndarray: The scaled keypoints.
    """
    keypoints = np.array(keypoints)
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return keypoints


def filter_invalid_keypoints(keypoints):
    """ Filters out invalid keypoints."""
    keypoints = np.array(keypoints)
    keypoints = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    return keypoints


def calculate_bounding_box(keypoints, frame_shape, margin=50):
    """
    Calculates the minimum and maximum coordinates of the bounding box for a set of keypoints.
    Applies stabilization to prevent jitter between frames.

    Parameters:
    - keypoints (list): The keypoints to calculate the bounding box for.
    - frame_shape (tuple): The dimensions of the frame.
    - margin (int): The margin to add to the bounding box.

    Returns:
    - tuple: The minimum and maximum coordinates of the bounding box.
    """
    keypoints = np.array(keypoints)
    
    # Handle empty keypoints array
    if keypoints.size == 0 or keypoints.shape[0] == 0:
        # Default to center of frame with reasonable size
        center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2
        half_width = min(100, frame_shape[1] // 4)
        half_height = min(100, frame_shape[0] // 4)
        
        return (
            max(0, center_x - half_width),
            max(0, center_y - half_height),
            min(frame_shape[1], center_x + half_width),
            min(frame_shape[0], center_y + half_height)
        )
    
    # Use first two columns (x,y) for coordinate calculation
    xy_points = keypoints[:, :2] if keypoints.shape[1] >= 2 else keypoints
    
    # Calculate center point of keypoints
    center_x, center_y = np.mean(xy_points, axis=0).astype(int)
    
    # Calculate the distance from center to farthest point
    distances = np.sqrt(np.sum((xy_points - np.array([center_x, center_y]))**2, axis=1))
    max_distance = int(np.max(distances)) if distances.size > 0 else margin
    
    # Use max distance plus margin for a more stable box size
    box_size = max_distance + margin
    
    # Calculate box coordinates based on center and size
    x_min = max(0, center_x - box_size)
    y_min = max(0, center_y - box_size)
    x_max = min(frame_shape[1], center_x + box_size)
    y_max = min(frame_shape[0], center_y + box_size)

    return int(x_min), int(y_min), int(x_max), int(y_max)


def interpolate_and_sort_df(df):
    """
    Interpolates missing values in the dataframe and sorts it.

    Parameters:
    - df (pd.DataFrame): The dataframe to interpolate and sort.

    Returns:
    - pd.DataFrame: The interpolated and sorted dataframe.
    """
    df = df.reset_index(drop=True)
    df.replace(0, np.nan, inplace=True)

    # Convert 'person_id' to numeric, set non-numeric entries to NaN
    df['person_id'] = pd.to_numeric(df['person_id'], errors='coerce')

    # Drop rows where 'person_id' is NaN (i.e., invalid person identifiers)
    df = df.dropna(subset=['person_id'])
    df['person_id'] = df['person_id'].astype(int)

    # Group the DataFrame by 'person_id'
    grouped_df = df.groupby('person_id')

    # Interpolate missing values linearly for each group
    df = grouped_df.apply(lambda group: group.interpolate(
        method='linear', limit_direction='both')).reset_index(drop=True)

    # Sort values by 'frame_number' and 'person_id'
    df.sort_values(['frame_number', 'person_id'], inplace=True)

    return df


def load_dataframe_from_csv(csv_path):
    """ Loads a dataframe from a CSV file. """
    dtype_spec = {'frame_number': int, 'x_0': float, 'y_0': float, 'x_1': float, 'y_1': float, 'x_2': float, 'y_2': float,
                  'x_3': float, 'y_3': float, 'x_4': float, 'y_4': float, 'x_5': float, 'y_5': float, 'x_6': float, 'y_6': float,
                  'x_7': float, 'y_7': float, 'x_8': float, 'y_8': float, 'x_9': float, 'y_9': float, 'x_10': float, 'y_10': float,
                  'x_11': float, 'y_11': float, 'x_12': float, 'y_12': float, 'x_13': float, 'y_13': float, 'x_14': float,
                  'y_14': float, 'x_15': float, 'y_15': float, 'x_16': float, 'y_16': float}
    try:
        df = pd.read_csv(csv_path, dtype=dtype_spec, low_memory=False)
        logging.info(f"Loaded dataframe from {csv_path}.")
        return df
    except FileNotFoundError:
        logging.error(
            f"CSV file not found at path: {csv_path}. Please ensure the file path is correct.")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"CSV file at {csv_path} is empty.")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file at {csv_path}: {e}")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            return df
        except Exception as e:
            logging.error(f"Failed to load with inferred dtypes: {e}")
            raise


def create_progress_bar(total, desc, show_progress):
    """Creates a progress bar using the tqdm library."""
    return tqdm(total=total, desc=desc, ncols=100) if show_progress else None
