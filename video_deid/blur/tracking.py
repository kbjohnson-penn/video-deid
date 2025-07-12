"""
Kalman filter tracking for face position estimation across video frames
"""
import logging
import numpy as np
import pandas as pd
import cv2
from ..utils import filter_invalid_keypoints
from ..config import FACIAL_KEYPOINTS_COUNT


def initialize_kalman_filter(fps):
    """
    Initializes a Kalman filter.

    Parameters:
    - fps (float): The frame rate.

    Returns:
    - cv2.KalmanFilter: The initialized Kalman filter.
    """
    dt = 1/fps

    kf = cv2.KalmanFilter(4, 2)  # 4 dynamic states, 2 measurement states

    # State Transition Matrix (A)
    kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

    # Measurement Matrix (H)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

    # Process Noise Covariance (Q)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    # Measurement Noise Covariance (R)
    kf.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 1

    # Error Covariance Matrix (P)
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    # Initial State (x)
    kf.statePost = np.zeros(4, np.float32)

    return kf


def initialize_kalman_filter_for_person(person_id, kalman_filters, keypoints, frame_width, frame_height, fps):
    """
    Initializes the Kalman filter for a given person if they do not have an existing filter.

    Parameters:
    - person_id (int): The identifier of the person.
    - kalman_filters (dict): Dictionary of Kalman filters.
    - keypoints (list): The keypoints to initialize the filter with.
    - frame_width (int): The width of the frame.
    - frame_height (int): The height of the frame.
    - fps (float): The frame rate.

    Returns:
    - dict: Updated dictionary of Kalman filters.
    """
    if person_id in kalman_filters:
        return kalman_filters

    if keypoints and len(keypoints) > 0:
        initial_keypoints = np.array(keypoints)
        initial_keypoints = filter_invalid_keypoints(initial_keypoints)
        if initial_keypoints.shape[0] > 0:
            initial_position = np.mean(initial_keypoints, axis=0)[:2]
            kf = initialize_kalman_filter(fps)
            kf.statePost[:2] = initial_position.astype(np.float32)
            kf.statePost[2:4] = [0, 0]
            kalman_filters[person_id] = kf
            logging.info(f"Initialized Kalman Filter for person {person_id}.")

    return kalman_filters


def kalman_filter_and_predict(keypoints, kalman_filters, person_id, frame_width, frame_height, frame_number, fps):
    """
    Applies a Kalman filter to a set of keypoints and predicts the next state.

    Parameters:
    - keypoints (list): The keypoints to filter.
    - kalman_filters (dict): Dictionary of Kalman filters.
    - person_id (int): The identifier of the person.
    - frame_width (int): The width of the frame.
    - frame_height (int): The height of the frame.
    - frame_number (int): The frame number.
    - fps (float): The frame rate.

    Returns:
    - tuple: (estimated_x, estimated_y, updated_kalman_filters)
    """
    # Early return if the Kalman filter doesn't exist and keypoints are not sufficient to initialize it.
    if person_id not in kalman_filters:
        kalman_filters = initialize_kalman_filter_for_person(
            person_id, kalman_filters, keypoints, frame_width, frame_height, fps)
        if person_id not in kalman_filters:
            logging.info(
                f"Skipping prediction for person {person_id} at frame {frame_number} due to insufficient keypoints.")
            return None, None, kalman_filters

    # Retrieve the Kalman filter
    kf = kalman_filters.get(person_id)
    if kf is None:
        logging.info(
            f"Kalman filter is None for person {person_id} at frame {frame_number}.")
        return None, None, kalman_filters

    # Predict the next state with the Kalman Filter
    predicted = kf.predict()

    # Update the Kalman filter if keypoints are available
    if keypoints and len(keypoints) > 0:
        keypoints = np.array(keypoints)
        keypoints = filter_invalid_keypoints(keypoints)
        if keypoints.shape[0] > 0:
            measurement = np.mean(keypoints, axis=0)[:2]
            kf.correct(measurement.astype(np.float32))

    # Extract predicted coordinates
    estimated_x, estimated_y = int(predicted[0]), int(predicted[1])
    return estimated_x, estimated_y, kalman_filters


def get_missing_keypoints_from_dataframe(df, frame_number, person_id):
    """
    Gets the missing keypoints for a frame from a dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe to get the missing keypoints from.
    - frame_number (int): The frame number.
    - person_id (int): The identifier of the person.

    Returns:
    - list: The missing keypoints.
    """
    rows = df[(df['frame_number'] == frame_number) &
              (df['person_id'] == person_id)]

    if rows.empty:
        return None

    # Extract the facial keypoints from the first row
    keypoints = [[rows[f'x_{i}'].values[0], rows[f'y_{i}'].values[0],
                  rows[f'c_{i}'].values[0]] for i in range(FACIAL_KEYPOINTS_COUNT)]

    return keypoints


def generate_kalman_predictions(keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames):
    """
    Generates predictions for face positions using Kalman filtering.

    Parameters:
    - keypoints_df (pd.DataFrame): Dataframe containing keypoints.
    - interpolated_keypoints_df (pd.DataFrame): Dataframe containing interpolated keypoints.
    - frame_width (int): The width of the frame.
    - frame_height (int): The height of the frame.
    - fps (float): The frame rate.
    - total_frames (int): The total number of frames in the video.

    Returns:
    - pd.DataFrame: Dataframe containing Kalman filter predictions.
    """
    # Convert grouped DataFrames to dictionaries for faster lookup
    keypoints_dict = {frame: group for frame,
                      group in keypoints_df.groupby('frame_number')}
    interpolated_dict = {frame: group for frame,
                         group in interpolated_keypoints_df.groupby('frame_number')}

    # Pre-compute person IDs
    all_person_ids = keypoints_df['person_id'].unique()

    # Initialize Kalman filters for all persons at once
    kalman_filters = {}
    for person_id in all_person_ids:
        person_keypoints = keypoints_df[keypoints_df['person_id'] == person_id]
        if not person_keypoints.empty:
            first_valid_row = person_keypoints.iloc[0]
            keypoints = [[first_valid_row[f'x_{i}'], first_valid_row[f'y_{i}'],
                         first_valid_row[f'c_{i}']] for i in range(FACIAL_KEYPOINTS_COUNT)]
            kalman_filters = initialize_kalman_filter_for_person(
                person_id, kalman_filters, keypoints, frame_width, frame_height, fps)

    results = []

    # Use vectorized operations where possible
    for frame_number in range(1, total_frames + 1):
        frame_data = keypoints_dict.get(frame_number, pd.DataFrame())

        if frame_data.empty:
            results.append({'frame_number': frame_number, 'person_id': None,
                           'estimated_x': None, 'estimated_y': None})
            continue

        for _, row in frame_data.iterrows():
            person_id = row['person_id']

            # Extract keypoints efficiently
            keypoints = np.array(
                [[row[f'x_{i}'], row[f'y_{i}'], row[f'c_{i}']] for i in range(FACIAL_KEYPOINTS_COUNT)])

            # Check if keypoints are valid using vectorized operations
            if np.all(keypoints[:, :2] == [0, 0]) and frame_number in interpolated_dict:
                interpolated_frame = interpolated_dict[frame_number]
                interp_row = interpolated_frame[interpolated_frame['person_id'] == person_id]
                if not interp_row.empty:
                    keypoints = np.array([[interp_row[f'x_{i}'].values[0],
                                           interp_row[f'y_{i}'].values[0],
                                           interp_row[f'c_{i}'].values[0]] for i in range(FACIAL_KEYPOINTS_COUNT)])

            # Predict position
            estimated_x, estimated_y, kalman_filters = kalman_filter_and_predict(
                keypoints.tolist(), kalman_filters, person_id, frame_width, frame_height, frame_number, fps
            )

            results.append({'frame_number': frame_number, 'person_id': person_id,
                           'estimated_x': estimated_x, 'estimated_y': estimated_y})

    return pd.DataFrame(results)
