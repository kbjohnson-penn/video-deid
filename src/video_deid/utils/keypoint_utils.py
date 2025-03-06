"""
Utility functions for working with keypoints in videos
"""
import logging
import numpy as np
import pandas as pd


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
    """
    Filters out invalid keypoints.
    
    Parameters:
    - keypoints (list or np.ndarray): Array of keypoints with x,y coordinates
    
    Returns:
    - np.ndarray: Filtered keypoints
    """
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
    - tuple: The minimum and maximum coordinates of the bounding box (x_min, y_min, x_max, y_max).
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
    """
    Loads a dataframe from a CSV file.
    
    Parameters:
    - csv_path (str): Path to the CSV file
    
    Returns:
    - pd.DataFrame: The loaded dataframe
    
    Raises:
    - FileNotFoundError: If the CSV file does not exist
    - pd.errors.EmptyDataError: If the CSV file is empty
    - pd.errors.ParserError: If there's an error parsing the CSV file
    """
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