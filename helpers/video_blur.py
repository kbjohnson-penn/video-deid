import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import cv2
from helpers.utils import get_video_properties, calculate_time, scale_keypoints, filter_invalid_keypoints, calculate_bounding_box


def apply_circular_blur(frame, x_min, y_min, x_max, y_max):
    """
    Applies a circular blur to a region in a frame.

    Parameters:
    - frame (np.array): The frame to apply the blur to.
    - x_min, y_min, x_max, y_max (int): The coordinates of the region to blur.

    Returns:
    - np.array: The frame with the blurred region.
    """
    # Calculate the center and radius of the region to blur
    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    radius = int(max((x_max - x_min) // 2, (y_max - y_min) // 2))

    radius = radius * 2

    # Create a mask of the same size as the frame
    mask = np.zeros(frame.shape[:2], dtype="uint8")

    # Draw a filled circle (i.e., a disk) on the mask at the center of the region to blur
    cv2.circle(mask, center, radius, 255, -1)

    # Apply a Gaussian blur to the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)

    # Use the mask to replace the region to blur in the original frame with the corresponding region in the blurred frame
    np.copyto(frame, blurred_frame, where=mask[:, :, None] == 255)

    return frame


def draw_bounding_box_and_keypoints(frame, keypoints, x_min, y_min, x_max, y_max):
    """
    Draws a bounding box and keypoints on a frame.

    Parameters:
    - frame (np.array): The frame to draw on.
    - keypoints (list): The keypoints to draw.
    - x_min, y_min, x_max, y_max (int): The coordinates of the bounding box.

    Returns:
    - np.array: The frame with the bounding box and keypoints drawn on it.
    """
    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw the keypoints on the frame
    for keypoint in keypoints:
        cv2.circle(frame, (int(keypoint[0]), int(
            keypoint[1])), 2, (0, 0, 255), -1)

    return frame


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
    # Get the rows for the specified frame number and person_id
    rows = df[(df['frame_number'] == frame_number) &
              (df['person_id'] == person_id)]

    if rows.empty:
        return None

    # If there are rows
    if not rows.empty:
        # Extract the facial keypoints from the first row
        keypoints = [[rows[f'x_{i}'].values[0], rows[f'y_{i}'].values[0],
                      rows[f'c_{i}'].values[0]] for i in range(5)]

        # Return the keypoints
        return keypoints


def log_keypoints_and_save_frame(frame, frame_number, fps, keypoints, missing_frames_dir):
    """
    Logs a warning message and saves the frame to a file if there are no facial keypoints.

    Parameters:
    - frame (np.array): The frame to save.
    - frame_number (int): The frame number.
    - fps (float): The frame rate.
    - keypoints (list): The keypoints.
    - missing_frames_dir (str): The directory to save the frame to.

    Returns:
    - None
    """

    time_in_video = calculate_time(frame_number, fps)
    # Log a warning message
    logging.warning(
        f"No facial keypoints found for frame {frame_number} at time {time_in_video} seconds. Keypoints: {keypoints}")

    # Define the output file name
    output_file_name = f"frame_{frame_number}.jpg"

    # Define the output file path
    output_file_path = os.path.join(missing_frames_dir, output_file_name)

    # Save the frame to the output file
    cv2.imwrite(output_file_path, frame)


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


def kalman_filter_and_predict(keypoints, kalman_filters, person_id, frame_width, frame_height, fps):
    """
    Applies a Kalman filter to a set of keypoints and predicts the next state.

    Parameters:
    - keypoints (list): The keypoints to apply the Kalman filter to.
    - kalman_filters (dict): A dictionary of Kalman filters.
    - person_id (int): The identifier of the person.
    - frame_width (int): The width of the frame.
    - frame_height (int): The height of the frame.
    - fps (float): The frame rate.

    Returns:
    - int: The estimated x-coordinate of the person.
    - int: The estimated y-coordinate of the person.
    - dict: The updated dictionary of Kalman filters.
    """
    kf = None
    if person_id not in kalman_filters:
        # Initialize Kalman Filter with the initial position if keypoints are available
        if keypoints and len(keypoints) > 0:
            initial_keypoints = np.array(keypoints)
            initial_keypoints = scale_keypoints(
                initial_keypoints, frame_width, frame_height)
            initial_keypoints = filter_invalid_keypoints(initial_keypoints)
            if initial_keypoints.shape[0] > 0:
                initial_position = np.mean(initial_keypoints, axis=0)[:2]
                kf = initialize_kalman_filter(fps)
                # Set the initial state with the position and zero velocity
                kf.statePost[:2] = initial_position.astype(np.float32)
                kf.statePost[2:4] = [0, 0]
                kalman_filters[person_id] = kf
        logging.info(
            f"Person {person_id} not in kalman_filters. Initializing Kalman Filter.")
    else:
        kf = kalman_filters[person_id]

    if kf is None:
        logging.info(
            f"Kalman filter object for person {person_id} is None. Skipping.")
        return None, None, kalman_filters

    # Predict the next state with the Kalman Filter
    predicted = kf.predict()

    # If keypoints are available, update the Kalman Filter
    if keypoints and len(keypoints) > 0:
        keypoints = np.array(keypoints)
        keypoints = scale_keypoints(keypoints, frame_width, frame_height)
        keypoints = filter_invalid_keypoints(keypoints)
        if keypoints.shape[0] > 0:
            measurement = np.mean(keypoints, axis=0)[:2]
            kf.correct(measurement.astype(np.float32))

    estimated_x, estimated_y = int(predicted[0]), int(predicted[1])

    return estimated_x, estimated_y, kalman_filters


def generate_kalman_predictions(keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames):
    """
    Generates Kalman predictions for a dataframe of keypoints.

    Parameters:
    - keypoints_df (pd.DataFrame): The dataframe of keypoints.
    - interpolated_keypoints_df (pd.DataFrame): The dataframe of interpolated keypoints.
    - frame_width (int): The width of the frames.
    - frame_height (int): The height of the frames.
    - fps (float): The frame rate.
    - total_frames (int): The total number of frames.

    Returns:
    - pd.DataFrame: The dataframe of Kalman predictions.
    """

    # Initialize a dictionary to store the Kalman filters
    kalman_filters = {}
    frame_number = 1

    # Initialize an empty DataFrame
    result_df = pd.DataFrame(
        columns=['frame_number', 'person_id', 'estimated_x', 'estimated_y'])
    # Iterate over each frame
    while True:
        # Get the rows for the current frame number
        current_frame = keypoints_df[keypoints_df['frame_number']
                                     == frame_number]

        # If there are no rows, break the loop
        if current_frame.empty and frame_number > total_frames:
            break
        elif current_frame.empty:
            frame_number += 1
            continue

        # Iterate through the keypoints dataframe and get the keypoints for the current frame number and person_id
        for index, row in current_frame.iterrows():
            # Extract the facial keypoints from the row
            keypoints = [[row[f'x_{i}'], row[f'y_{i}'],
                          row[f'c_{i}']] for i in range(5)]
            # If are zeros in facial keypoints get the missing keypoints from the interpolated dataframe
            if all(keypoint[:2] == [0, 0] for keypoint in keypoints):
                # Get the missing keypoints from the interpolated dataframe
                logging.info(
                    f"Missing keypoints for frame {frame_number} and person {row['person_id']}. Getting from interpolated dataframe.")
                keypoints = get_missing_keypoints_from_dataframe(
                    interpolated_keypoints_df, frame_number, row['person_id'])

            # Get the estimated x and y coordinates and update the Kalman filters
            estimated_x, estimated_y, kalman_filters = kalman_filter_and_predict(
                keypoints, kalman_filters, row['person_id'], frame_width, frame_height, fps)

            # Create a new DataFrame for the current row and concatenate it with the result DataFrame
            current_df = pd.DataFrame([{'frame_number': frame_number, 'person_id': row['person_id'],
                                      'estimated_x': estimated_x, 'estimated_y': estimated_y}])
            result_df = pd.concat([result_df, current_df], ignore_index=True)

        frame_number += 1

    return result_df


def process_frame(frame, estimated_x, estimated_y, blur_faces=False, draw_bbox=False):
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

    # Draw bounding box based on smoothed or predicted position
    if blur_faces or draw_bbox:
        x_min, y_min, x_max, y_max = calculate_bounding_box(
            np.array([[estimated_x, estimated_y]]), frame.shape, margin=50)
        if blur_faces:
            frame = apply_circular_blur(frame, x_min, y_min, x_max, y_max)
        if draw_bbox:
            frame = draw_bounding_box_and_keypoints(
                frame, [[estimated_x, estimated_y]], x_min, y_min, x_max, y_max)
    return frame


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

    # Get the dimensions of the video frames, frame rate and total number of frames
    frame_width, frame_height, fps, total_frames = get_video_properties(
        video_path)
    logging.info('Getting video properties.')

    # Generate Kalman predictions
    logging.info('Generating Kalman predictions')
    predicted_df = generate_kalman_predictions(
        keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames)
    predicted_df.to_csv(kalman_filtered_keypoints_df_path, index=False)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    # Create a progress bar
    if show_progress:
        progress_bar = tqdm(total=total_frames,
                            desc="Processing video", ncols=100)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    # Initialize frame number
    frame_number = 1

    logging.info('Processing video...')
    # Process each frame in the video
    while cap.isOpened():
        # Update the progress bar if show_progress is True
        if show_progress:
            progress_bar.update(1)

        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # If no frame is returned, break the loop

        # Create a copy of the frame to apply changes
        frame_copy = frame.copy()

        # Iterate through the keypoints dataframe and get the keypoints for the current frame number and person_id
        for index, row in predicted_df[(predicted_df['frame_number'] == frame_number)].iterrows():
            try:
                # Process the frame
                process_frame(
                    frame_copy, row['estimated_x'], row['estimated_y'], blur_faces=False, draw_bbox=True)
            except Exception as e:
                logging.error(f"Error processing frame {frame_number}: {e}")
                continue  # Skip to the next iteration of the loop

        # After the loop, assign the processed frame_copy back to frame
        frame = frame_copy

        # If the frame is valid
        if frame is not None:
            # Write the frame to the output video
            out.write(frame)
        else:  # If the frame is invalid
            logging.warning(f"Invalid frame: {frame_number}")

        frame_number += 1

    # Close the progress bar if show_progress is True
    if show_progress:
        progress_bar.close()

    # Close the video file, video writer and destroy all windows.
    cap.release()
    out.release()
    cv2.destroyAllWindows()
