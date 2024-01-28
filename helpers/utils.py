import os
import logging
import cv2
import numpy as np
import pandas as pd
import ast


def setup_logging(log_file=False):
    """
    Set up logging.

    Parameters:
    log_file (str): Path to the log file.

    Returns:
    None
    """
    # Set up logging
    if log_file:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(filename=log_file,
                            level=logging.INFO, format=log_format)


def make_directory(path):
    """
    Creates a directory.

    Parameters:
    path (str): Path to the directory to create.

    Returns:
    str: The path to the created directory.
    """

    # Create the directory
    os.makedirs(path, exist_ok=True)

    # Return the new directory path
    return path


def get_video_properties(video_path):
    """
    Extracts the dimensions of the video frames, frame rate and total number of frames.

    Parameters:
    video_path (str): Path to the video file.

    Returns:
    tuple: The frame width, frame height, frame rate and total number of frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    # Get the dimensions of the video frames, frame rate and total number of frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Close the video file
    cap.release()

    return frame_width, frame_height, fps, total_frames


def calculate_time(frame_number, frame_rate):
    """
    Calculates the time in the video given the frame number and the frame rate.

    Parameters:
    frame_number (int): The frame number.
    frame_rate (float): The frame rate of the video.

    Returns:
    float: The time in the video.
    """
    return frame_number / frame_rate


def scale_keypoints(keypoints, width, height):
    """
    Scales keypoints to the frame dimensions.

    Parameters:
    keypoints (list): The keypoints to scale.
    width (int): The width of the frame.
    height (int): The height of the frame.

    Returns:
    list: The scaled keypoints.
    """

    # Convert keypoints to a numpy array
    keypoints = np.array(keypoints)

    # Assuming keypoints are normalized (i.e., in the range [0, 1])
    # Scale the keypoints to the frame dimensions
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return keypoints


def filter_invalid_keypoints(keypoints):
    """
    Filters out invalid keypoints.

    Parameters:
    keypoints (list): The keypoints to filter.

    Returns:
    list: The filtered keypoints.
    """
    # Filter out invalid keypoints
    # Keypoints are invalid if any coordinate is negative or if they are (0,0)
    keypoints = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    return keypoints


def calculate_bounding_box(keypoints, frame_shape, margin=50):
    """
    Calculates the minimum and maximum coordinates of the bounding box for a set of keypoints.

    Parameters:
    keypoints (list): The keypoints to calculate the bounding box for.
    frame_shape (tuple): The dimensions of the frame.
    margin (int): The margin to add to the bounding box.

    Returns:
    tuple: The minimum and maximum coordinates of the bounding box.
    """
    # Calculate the minimum and maximum coordinates of the bounding box
    x_min, y_min = np.min(keypoints[:, :2], axis=0).astype(int) - margin
    x_max, y_max = np.max(keypoints[:, :2], axis=0).astype(int) + margin

    # Ensure the bounding box coordinates are within the frame dimensions
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_shape[1], x_max)
    y_max = min(frame_shape[0], y_max)

    return x_min, y_min, x_max, y_max


def interpolate_keypoints(df):
    """
    Applies linear interpolation on the frames with missing keypoints.

    Parameters:
    df (pd.DataFrame): The dataframe to interpolate the keypoints for.

    Returns:
    pd.DataFrame: The dataframe with the interpolated keypoints.
    """
    df = df.reset_index(drop=True)

    # Replace zeros with NaNs
    df.replace(0, np.nan, inplace=True)

    # Group the dataframe by the person_id
    grouped_df = df.groupby('person_id')

    # Apply interpolation to each group and reset the index
    df = grouped_df.apply(lambda group: group.interpolate(
        method='linear', limit_direction='both')).reset_index(drop=True)

    return df


# Function to sort the df by frame_number and person_id
def sort_df(df):
    """
    Sorts the dataframe by frame_number and person_id.

    Parameters:
    df (pd.DataFrame): The dataframe to sort.

    Returns:
    pd.DataFrame: The sorted dataframe.
    """
    # Sort the dataframe by frame_number and person_id
    df.sort_values(['frame_number', 'person_id'], inplace=True)

    # Return the sorted dataframe
    return df


def interpolate_and_sort_df(df):
    """
    Applies linear interpolation and sorts the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe to interpolate and sort.

    Returns:
    pd.DataFrame: The interpolated and sorted dataframe.
    """
    # Interpolate the dataframe
    df = interpolate_keypoints(df)

    # Sort the dataframe
    df = sort_df(df)

    # Return the dataframe
    return df


def read_keypoints_from_csv(file_path):
    """
    Reads the keypoints from a file.

    Parameters:
    file_path (str): The path to the file containing the keypoints.

    Returns:
    list: The keypoints.
    """

    keypoints_file = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert string representation of list to an actual list
            keypoints_file.append(ast.literal_eval(line.strip()))

    keypoints_list = []
    for keypoints in keypoints_file:
        if len(keypoints) == 17:
            keypoints = [keypoints]
        keypoints_list.append(keypoints)

    return keypoints_list


# Function to create a dataframe from the list of keypoints. If there are multiple people in the frame, the keypoints for each person are stored in a separate row. The dataframe is sorted by frame_number and person_id.
def create_keypoints_dataframe(keypoints_list):
    """
    Creates a dataframe from the list of keypoints. If there are multiple people in the frame, the keypoints for each person are stored in a separate row. The dataframe is sorted by frame_number and person_id.

    Parameters:
    keypoints_list (list): The list of keypoints.

    Returns:
    pd.DataFrame: The dataframe containing the keypoints.
    """

    df_list = []
    for frame_number, keypoints in enumerate(keypoints_list):
        for person_id, keypoint_coordinates in enumerate(keypoints):
            person_data = {
                'frame_number': frame_number,
                'person_id': person_id,
            }
            for i, (x, y, c) in enumerate(keypoint_coordinates):
                person_data.update({
                    f'x_{i}': x,
                    f'y_{i}': y,
                    f'c_{i}': c
                })
            df_list.append(person_data)

    df = pd.DataFrame(df_list)
    df = df.sort_values(by=['frame_number', 'person_id'])

    return df
