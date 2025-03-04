import numpy as np
import pandas as pd
import logging
import cv2
from pathlib import Path
from helpers.utils import get_video_properties, calculate_time, filter_invalid_keypoints, calculate_bounding_box, create_progress_bar


def apply_circular_blur(frame, x_min, y_min, x_max, y_max):
    """
    Applies a circular blur to a region in a frame with feathered edges for smoother transition.

    Parameters:
    - frame (np.array): The frame to apply the blur to.
    - x_min, y_min, x_max, y_max (int): The coordinates of the region to blur.

    Returns:
    - np.array: The frame with the blurred region.
    """
    # Make a copy of the frame to avoid modifying the original
    result = frame.copy()
    
    # Calculate center and radius
    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    radius = int(max((x_max - x_min) // 2, (y_max - y_min) // 2))
    
    # Use slightly larger radius for the blur area
    radius = radius * 2
    
    # Create a mask for the blur region
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    
    # Create a feathered edge mask for smooth transition
    feather_size = min(30, radius // 4)  # Feather size relative to radius
    if feather_size > 0:
        mask_feathered = cv2.GaussianBlur(mask.astype(np.float32), 
                                         (feather_size*2+1, feather_size*2+1), 
                                         feather_size)
        mask_feathered = (mask_feathered * 255).astype(np.uint8)
    else:
        mask_feathered = mask
    
    # Apply stronger blur for better de-identification
    # Use kernel size proportional to the radius for more consistent blurring
    kernel_size = max(99, min(199, 2 * radius + 1))
    # Make kernel size odd if it's even
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    
    # Apply stronger blur to completely obscure features
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), kernel_size/3)
    
    # Create normalized alpha mask (0.0 to 1.0) for blending
    alpha = mask_feathered.astype(float) / 255.0
    
    # Apply alpha blending for each channel
    for c in range(3):  # RGB channels
        result[:,:,c] = frame[:,:,c] * (1 - alpha) + blurred[:,:,c] * alpha
        
    return result


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

    # Process Noise Covariance (Q) - Reduced to make prediction more stable
    # The diagonal elements represent uncertainty in position and velocity
    process_noise = np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ], np.float32) * 0.01  # Reduced process noise factor
    
    kf.processNoiseCov = process_noise

    # Measurement Noise Covariance (R) - Increased to trust measurements less
    # This will make the filter less reactive to noisy measurements
    kf.measurementNoiseCov = np.array([[10, 0],
                                       [0, 10]], np.float32)

    # Error Covariance Matrix (P)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 10  # Initial uncertainty

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
    Uses confidence-weighted keypoint averaging and adaptive measurement noise.
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
        valid_keypoints = filter_invalid_keypoints(keypoints)
        
        if valid_keypoints.shape[0] > 0:
            # Get confidence values for valid keypoints
            confidences = valid_keypoints[:, 2]
            
            # Normalize confidences to use as weights (if all confidences are 0, use equal weights)
            if np.sum(confidences) > 0:
                weights = confidences / np.sum(confidences)
                # Weighted average of keypoints based on confidence
                measurement = np.average(valid_keypoints[:, :2], axis=0, weights=weights)
            else:
                # Use simple average if no confidence information
                measurement = np.mean(valid_keypoints[:, :2], axis=0)
            
            # Adjust measurement noise based on average confidence
            # Lower confidence = higher measurement noise (trust measurements less)
            avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0
            if avg_confidence > 0:
                # Scale measurement noise inversely with confidence
                # High confidence (near 1.0) = low noise (near 5)
                # Low confidence (near 0.0) = high noise (near 50)
                noise_scale = 5 + (1.0 - avg_confidence) * 45
                kf.measurementNoiseCov = np.array([[noise_scale, 0], 
                                                  [0, noise_scale]], np.float32)
            
            # Update filter with measurement
            kf.correct(measurement.astype(np.float32))
            
            # Use velocity damping for smoother trajectories
            # Reduce velocity component if prediction jumped too far
            current_velocity = kf.statePost[2:4]
            velocity_magnitude = np.linalg.norm(current_velocity)
            
            # Apply velocity dampening if the velocity is too high
            max_velocity = 20  # max pixels per frame
            if velocity_magnitude > max_velocity:
                damping_factor = max_velocity / velocity_magnitude
                kf.statePost[2:4] *= damping_factor

    # Extract predicted coordinates with bounds checking
    estimated_x = max(0, min(frame_width - 1, int(kf.statePost[0])))
    estimated_y = max(0, min(frame_height - 1, int(kf.statePost[1])))
    
    return estimated_x, estimated_y, kalman_filters


def generate_kalman_predictions(keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames):
    """
    Generates Kalman predictions for all frames, including those without keypoints, to maintain synchronization.
    
    Parameters:
    - keypoints_df: DataFrame with original keypoints data
    - interpolated_keypoints_df: DataFrame with interpolated keypoints for missing frames
    - frame_width, frame_height: Video frame dimensions
    - fps: Frames per second of the video
    - total_frames: Total number of frames in video
    
    Returns:
    - DataFrame with Kalman filter predictions for each frame/person
    """
    # Initialize a dictionary to store the Kalman filters
    kalman_filters = {}

    # Pre-compute grouped DataFrames - this is a performance optimization
    # as we avoid grouping operations inside the loop
    grouped_keypoints_df = keypoints_df.groupby('frame_number')
    grouped_interpolated_df = interpolated_keypoints_df.groupby('frame_number')
    
    # Get the set of frame numbers that exist in the input data for faster lookups
    keypoints_frames = set(grouped_keypoints_df.groups.keys())
    interpolated_frames = set(grouped_interpolated_df.groups.keys())

    # Pre-allocate memory for results
    results = []
    results_capacity = total_frames * (len(keypoints_df['person_id'].unique()) or 1)
    results = [None] * results_capacity
    result_index = 0

    # Process each frame in the video
    for frame_number in range(1, total_frames + 1):
        # Check if frame has keypoints data with fast set lookup
        if frame_number in keypoints_frames:
            current_frame = grouped_keypoints_df.get_group(frame_number)
            
            # Process each person in this frame
            for _, row in current_frame.iterrows():
                person_id = row['person_id']
                
                # Get keypoints more efficiently - extract facial keypoints (first 5)
                # This avoids constructing list comprehension for each row
                keypoints = []
                all_zeros = True
                
                for i in range(5):
                    x, y, c = row[f'x_{i}'], row[f'y_{i}'], row[f'c_{i}']
                    keypoints.append([x, y, c])
                    if not (x == 0 and y == 0):
                        all_zeros = False
                
                # Use interpolated data if original keypoints are all zeros
                if all_zeros and frame_number in interpolated_frames:
                    interpolated_row = grouped_interpolated_df.get_group(frame_number)
                    interpolated_keypoints = get_missing_keypoints_from_dataframe(
                        interpolated_row, frame_number, person_id)
                    if interpolated_keypoints:
                        keypoints = interpolated_keypoints
                
                # Get Kalman predictions
                estimated_x, estimated_y, kalman_filters = kalman_filter_and_predict(
                    keypoints, kalman_filters, person_id, frame_width, frame_height, frame_number, fps
                )
                
                # Store result
                if result_index < results_capacity:
                    results[result_index] = {
                        'frame_number': frame_number,
                        'person_id': person_id, 
                        'estimated_x': estimated_x, 
                        'estimated_y': estimated_y
                    }
                    result_index += 1
                else:
                    # If we exceed pre-allocated capacity, append
                    results.append({
                        'frame_number': frame_number,
                        'person_id': person_id, 
                        'estimated_x': estimated_x, 
                        'estimated_y': estimated_y
                    })
        else:
            # For frames with no keypoints data, add a placeholder row
            if result_index < results_capacity:
                results[result_index] = {
                    'frame_number': frame_number,
                    'person_id': None,
                    'estimated_x': None, 
                    'estimated_y': None
                }
                result_index += 1
            else:
                results.append({
                    'frame_number': frame_number,
                    'person_id': None,
                    'estimated_x': None, 
                    'estimated_y': None
                })

    # Remove unused pre-allocated entries and create DataFrame
    results = [r for r in results if r is not None]
    result_df = pd.DataFrame(results)
    
    # Log summary statistics
    person_count = len(result_df['person_id'].dropna().unique())
    logging.info(f"Generated Kalman predictions for {person_count} people across {total_frames} frames")
    
    return result_df


def process_frame(frame, estimated_x, estimated_y, blur_faces=False, draw_bbox=True):
    """
    Processes a frame and applies blur to faces.
    Uses stable bounding box calculation to reduce jitter.

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
        # Use larger margin (80) for blurring to ensure full face coverage
        x_min, y_min, x_max, y_max = calculate_bounding_box(
            np.array([[estimated_x, estimated_y]]), frame.shape, margin=80)

        # Add padding to ensure consistent box size between frames
        # This helps reduce jitter in the blurring
        width = x_max - x_min
        height = y_max - y_min
        
        # Ensure the box is square for more stable visual effect
        max_dim = max(width, height)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Recalculate box dimensions from center to ensure it's square
        x_min = max(0, center_x - max_dim // 2)
        y_min = max(0, center_y - max_dim // 2)
        x_max = min(frame.shape[1], center_x + max_dim // 2)
        y_max = min(frame.shape[0], center_y + max_dim // 2)

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
    - video_path (str or Path): Path to the input video.
    - output_path (str or Path): Path to the output video.
    - fps (float): Frame rate of the video.

    Returns:
    - cv2.VideoWriter: Video writer object.
    """
    # Convert to Path objects
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get video properties and initialize writer
    frame_width, frame_height, _, _ = get_video_properties(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Convert back to string for OpenCV
    return cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))


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
    # Pre-compute index for faster frame lookup
    frame_data_lookup = {}
    for frame_num, group in predicted_df.groupby('frame_number'):
        valid_rows = group[pd.notna(group['estimated_x']) & pd.notna(group['estimated_y'])]
        if not valid_rows.empty:
            frame_data_lookup[frame_num] = valid_rows
    
    # Create progress bar
    progress_bar = create_progress_bar(total_frames, "Processing and blurring video", show_progress)
    
    # Track processed frames for statistics
    processed_count = 0
    frame_number = 1
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we need to process this frame
        if frame_number in frame_data_lookup:
            current_frame_data = frame_data_lookup[frame_number]
            
            # Only make a copy if we're actually going to modify the frame
            if len(current_frame_data) > 0:
                frame_copy = frame.copy()
                
                # Process all detections in this frame
                for _, row in current_frame_data.iterrows():
                    try:
                        frame_copy = process_frame(
                            frame_copy, 
                            int(row['estimated_x']), 
                            int(row['estimated_y']), 
                            blur_faces=True, 
                            draw_bbox=False
                        )
                        processed_count += 1
                    except Exception as e:
                        logging.error(f"Error processing frame {frame_number}, person {row['person_id']}: {e}")
                        continue
                
                # Write processed frame
                out.write(frame_copy)
            else:
                out.write(frame)
        else:
            # No processing needed for this frame
            out.write(frame)

        # Update progress and counter
        frame_number += 1
        if show_progress:
            progress_bar.update(1)
            
        # Periodically log progress for long videos
        if frame_number % 500 == 0:
            logging.info(f"Processed {frame_number}/{total_frames} frames ({frame_number/total_frames*100:.1f}%)")

    # Cleanup
    if show_progress:
        progress_bar.close()
        
    # Log final stats
    logging.info(f"Video processing complete: {processed_count} faces blurred across {frame_number-1} frames")


def process_video(video_path, keypoints_df, interpolated_keypoints_df, kalman_filtered_keypoints_df_path, output_path, show_progress):
    """
    Processes a video based on the kalman filter predictions and saves the output to a file.

    Parameters:
    - video_path (str or Path): Path to the input video.
    - keypoints_df (pd.DataFrame): Dataframe containing keypoints.
    - interpolated_keypoints_df (pd.DataFrame): Dataframe containing interpolated keypoints.
    - kalman_filtered_keypoints_df_path (str or Path): Path to the Kalman filtered keypoints dataframe.
    - output_path (str or Path): Path to the output video.
    - show_progress (bool): Show progress bar.

    Returns:
    - None
    """
    # Convert to Path objects
    video_path = Path(video_path)
    kalman_filtered_keypoints_df_path = Path(kalman_filtered_keypoints_df_path)
    output_path = Path(output_path)
    
    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kalman_filtered_keypoints_df_path.parent.mkdir(parents=True, exist_ok=True)
    
    cap = None
    out = None
    
    try:
        # Get video properties
        frame_width, frame_height, fps, total_frames = get_video_properties(video_path)
        logging.info(f'Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames')

        # Generate Kalman predictions
        logging.info('Generating Kalman predictions')
        predicted_df = generate_kalman_predictions(
            keypoints_df, interpolated_keypoints_df, frame_width, frame_height, fps, total_frames)
        predicted_df.to_csv(str(kalman_filtered_keypoints_df_path), index=False)
        logging.info(f'Kalman predictions saved to: {kalman_filtered_keypoints_df_path}')

        # Initialize video capture and writer
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        out = initialize_video_writer(video_path, output_path, fps)
        logging.info('Starting video processing...')

        # Process frames
        process_frame_loop(cap, predicted_df, out, show_progress, total_frames)
        logging.info(f'Video processing complete. Output saved to: {output_path}')

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except IOError as e:
        logging.error(f"IO error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred in process_video: {e}")
        raise
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.debug(f"Could not destroy OpenCV windows: {e}")
            # Continue processing even if window destruction fails
