from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import cv2
from helpers.utils import get_video_properties, calculate_time, filter_invalid_keypoints, calculate_bounding_box, create_progress_bar


def apply_circular_blur(frame, x_min, y_min, x_max, y_max):
    """
    Applies a circular blur to a region in a frame.

    Parameters:
    - frame (np.array): The frame to apply the blur to.
    - x_min, y_min, x_max, y_max (int): The coordinates of the region to blur.

    Returns:
    - np.array: The frame with the blurred region.
    """
    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    radius = int(max((x_max - x_min) // 2, (y_max - y_min) // 2))

    radius = radius * 2

    mask = np.zeros(frame.shape[:2], dtype="uint8")

    cv2.circle(mask, center, radius, 255, -1)

    blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)

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
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

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
    rows = df[(df['frame_number'] == frame_number) &
              (df['person_id'] == person_id)]

    if rows.empty:
        return None

    # Extract the facial keypoints from the first row
    keypoints = [[rows[f'x_{i}'].values[0], rows[f'y_{i}'].values[0],
                  rows[f'c_{i}'].values[0]] for i in range(5)]

    return keypoints


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
    """Initializes the Kalman filter for a given person if they do not have an existing filter."""
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


def generate_kalman_predictions(keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames):
    """
    Generates Kalman predictions for all frames, including those without keypoints, to maintain synchronization.
    """
    # Initialize a dictionary to store the Kalman filters
    kalman_filters = {}

    # Group the DataFrame by frame number to optimize frame access
    grouped_keypoints_df = keypoints_df.groupby('frame_number')
    grouped_interpolated_df = interpolated_keypoints_df.groupby('frame_number')

    # List to hold results and avoid repeated DataFrame concatenation
    results = []

    # Iterate over each frame
    for frame_number in range(1, total_frames + 1):
        # Get the rows for the current frame number
        current_frame = grouped_keypoints_df.get_group(
            frame_number) if frame_number in grouped_keypoints_df.groups else pd.DataFrame()

        if current_frame.empty:
            # If there are no keypoints, still add a placeholder with None values for estimated_x and estimated_y
            results.append({'frame_number': frame_number, 'person_id': None,
                           'estimated_x': None, 'estimated_y': None})
            continue

        # Iterate through the keypoints dataframe and get the keypoints for the current frame number and person_id
        for _, row in current_frame.iterrows():
            # Extract the facial keypoints from the row
            keypoints = [[row[f'x_{i}'], row[f'y_{i}'],
                          row[f'c_{i}']] for i in range(5)]

            # If all keypoints are zeros, retrieve from interpolated dataframe
            if all(keypoint[:2] == [0, 0] for keypoint in keypoints) and frame_number in grouped_interpolated_df.groups:
                logging.info(
                    f"Missing keypoints for frame {frame_number} and person {row['person_id']} at time: {calculate_time(frame_number, fps)} seconds. Using interpolated dataframe."
                )
                interpolated_row = grouped_interpolated_df.get_group(
                    frame_number)
                keypoints = get_missing_keypoints_from_dataframe(
                    interpolated_row, frame_number, row['person_id'])

            # Predict the next state
            estimated_x, estimated_y, kalman_filters = kalman_filter_and_predict(
                keypoints, kalman_filters, row['person_id'], frame_width, frame_height, frame_number, fps
            )

            # Append results to the list
            results.append({'frame_number': frame_number,
                            'person_id': row['person_id'], 'estimated_x': estimated_x, 'estimated_y': estimated_y})

    # Convert results to DataFrame after processing all frames
    result_df = pd.DataFrame(results)

    return result_df


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
    if estimated_x is None or estimated_y is None:
        logging.warning(f"Skipping frame as keypoints are None.")
        return frame

    if draw_bbox or blur_faces:
        x_min, y_min, x_max, y_max = calculate_bounding_box(
            np.array([[estimated_x, estimated_y]]), frame.shape, margin=50)

        if blur_faces:
            frame = apply_circular_blur(frame, x_min, y_min, x_max, y_max)
        if draw_bbox:
            frame = draw_bounding_box_and_keypoints(
                frame, [[estimated_x, estimated_y]], x_min, y_min, x_max, y_max)

    return frame


def initialize_video_writer(video_path, output_path, fps):
    """
    Initializes a video writer object.

    Parameters:
    - video_path (str): Path to the input video.
    - output_path (str): Path to the output video.
    - fps (float): Frame rate of the video.

    Returns:
    - cv2.VideoWriter: Video writer object.
    """
    frame_width, frame_height, _, _ = get_video_properties(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


def process_frame_loop(cap, predicted_df, out, show_progress, total_frames):
    """
    Processes frames in a video and applies Gaussian blur to faces.

    Parameters:
    - cap (cv2.VideoCapture): Video capture object.
    - predicted_df (pd.DataFrame): Dataframe containing Kalman filter predictions.
    - out (cv2.VideoWriter): Video writer object.
    - show_progress (bool): Show progress bar.
    - total_frames (int): Total number of frames in the video.

    Returns:
    - None
    """
    frame_number = 1
    progress_bar =  create_progress_bar(total_frames, "Processing and blurring video", show_progress)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_data = predicted_df[predicted_df['frame_number']
                                          == frame_number]

        if not current_frame_data.empty:
            frame_copy = frame.copy()
            for _, row in current_frame_data.iterrows():
                if pd.notna(row['estimated_x']) and pd.notna(row['estimated_y']):
                    try:
                        logging.info(
                            f"Processing frame {frame_number} for person_id {row['person_id']}")
                        frame_copy = process_frame(
                            frame_copy, row['estimated_x'], row['estimated_y'], blur_faces=True, draw_bbox=False
                        )
                    except Exception as e:
                        logging.error(
                            f"Error processing frame {frame_number}: {e}")
                        continue

            out.write(frame_copy)
        else:
            out.write(frame)

        frame_number += 1
        if show_progress:
            progress_bar.update(1)

    if show_progress:
        progress_bar.close()


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
    try:
        frame_width, frame_height, fps, total_frames = get_video_properties(
            video_path)
        logging.info('Getting video properties.')

        # Generate Kalman predictions
        logging.info('Generating Kalman predictions')
        predicted_df = generate_kalman_predictions(
            keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames)
        predicted_df.to_csv(kalman_filtered_keypoints_df_path, index=False)

        # Initialize video capture and writer
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        out = initialize_video_writer(video_path, output_path, fps)
        logging.info('Starting video processing...')

        # Process frames
        process_frame_loop(cap, predicted_df, out, show_progress, total_frames)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except IOError as e:
        logging.error(f"IO error occurred: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred in process_video: {e}")

    finally:
        if cap:
            cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
