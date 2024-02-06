import os
from tqdm import tqdm
import numpy as np
import logging
from collections import defaultdict, deque
import cv2
from helpers.utils import get_video_properties, calculate_time, scale_keypoints, filter_invalid_keypoints, calculate_bounding_box


def apply_circular_blur(frame, x_min, y_min, x_max, y_max):
    """
    Applies a circular blur to a region in a frame.

    Parameters:
    frame (np.array): The frame to apply the blur to.
    x_min, y_min, x_max, y_max (int): The coordinates of the region to blur.

    Returns:
    np.array: The frame with the blurred region.
    """
    # Calculate the center and radius of the region to blur
    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    radius = int(max((x_max - x_min) // 2, (y_max - y_min) // 2))

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
    frame (np.array): The frame to draw on.
    keypoints (list): The keypoints to draw.
    x_min, y_min, x_max, y_max (int): The coordinates of the bounding box.

    Returns:
    np.array: The frame with the bounding box and keypoints drawn on it.
    """
    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw the keypoints on the frame
    for keypoint in keypoints:
        cv2.circle(frame, (int(keypoint[0]), int(
            keypoint[1])), 2, (0, 0, 255), -1)

    return frame


previous_bboxes = defaultdict(lambda: deque(maxlen=5))


def process_frame(frame, keypoints, person_id, blur_faces=False, draw_bbox=False):
    """
    Processes a frame: draws a bounding box and keypoints or applies a blur to the face region.

    Parameters:
    frame (np.array): The frame to process.
    keypoints (list): The keypoints to process.
    person_id (int): The identifier of the person.
    blur_faces (bool): Whether to blur the faces.
    draw_bbox (bool): Whether to draw a bounding box.

    Returns:
    np.array: The processed frame.
    """
    if frame is None:
        logging.error('Frame is None. Please check the video file.')
        return None

    global previous_bboxes  # Access the global variable that stores previous bounding boxes

    keypoints = np.array(keypoints)  # Convert keypoints to numpy array

    # Initialize bounding box variables
    x_min, y_min, x_max, y_max = None, None, None, None

    # If there are keypoints
    if keypoints.shape[0] > 0:

        # Scale keypoints according to frame dimensions
        keypoints = scale_keypoints(
            keypoints, frame.shape[1], frame.shape[0])

        # Filter out invalid keypoints
        keypoints = filter_invalid_keypoints(keypoints)

        margin = 50  # Define a margin to increase the size of the bounding box

        # If there are valid keypoints
        if keypoints.shape[0] > 0:
            x_min, y_min, x_max, y_max = calculate_bounding_box(
                keypoints, frame.shape, margin)

        # If bounding box variables have been assigned
        if None not in [x_min, y_min, x_max, y_max]:

            # Add the current bounding box to the list of previous bounding boxes for this person
            previous_bboxes[person_id].append((x_min, y_min, x_max, y_max))

            # Calculate the average bounding box coordinates for this person
            x_min, y_min, x_max, y_max = np.mean(
                previous_bboxes[person_id], axis=0).astype(int)

            if blur_faces:
                # Apply circular blur to the region defined by the bounding box
                frame = apply_circular_blur(
                    frame, x_min, y_min, x_max, y_max)
            if draw_bbox:
                # Draw the bounding box on the frame
                frame = draw_bounding_box_and_keypoints(
                    frame, keypoints, x_min, y_min, x_max, y_max)

            return frame


def get_missing_keypoints_from_dataframe(df, frame_number, person_id):
    """
    Gets the missing keypoints for a frame from a dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe to get the missing keypoints from.
    frame_number (int): The frame number.
    person_id (int): The identifier of the person.

    Returns:
    list: The missing keypoints.
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
    frame (np.array): The frame to save.
    frame_number (int): The frame number.
    fps (float): The frame rate.
    keypoints (list): The keypoints.
    missing_frames_dir (str): The directory to save the frame to.

    Returns:
    None
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


def process_video(video_path, keypoints_df, interpolated_keypoints_df, output_path, missing_frames_dir, show_progress):
    """
    Process the video and apply blur to faces.

    Parameters:
    video_path (str): Path to the input video.
    keypoints_df (pd.DataFrame): Dataframe containing keypoints.
    interpolated_keypoints_df (pd.DataFrame): Dataframe containing interpolated keypoints.
    output_path (str): Path to the output video.
    missing_frames_dir (str): Directory to save frames that don't have keypoints.
    show_progress (bool): Show progress bar.

    Returns:
    None
    """

    # Get the dimensions of the video frames, frame rate and total number of frames
    frame_width, frame_height, fps, total_frames = get_video_properties(
        video_path)
    logging.info('Getting video properties.')

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
        for index, row in keypoints_df[(keypoints_df['frame_number'] == frame_number)].iterrows():
            # Extract the facial keypoints from the row
            keypoints = [[row[f'x_{i}'], row[f'y_{i}'],
                          row[f'c_{i}']] for i in range(5)]

            # If are zeros in facial keypoints get the missing keypoints from the interpolated dataframe
            if all(keypoint[:2] == [0, 0] for keypoint in keypoints):
                # If there are no missing keypoints
                log_keypoints_and_save_frame(
                    frame, frame_number, fps, keypoints, missing_frames_dir)

                # Get the missing keypoints from the interpolated dataframe
                keypoints = get_missing_keypoints_from_dataframe(
                    interpolated_keypoints_df, frame_number, row['person_id'])
            try:
                frame_copy = process_frame(
                    frame_copy, keypoints, row['person_id'], blur_faces=True, draw_bbox=False)

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
