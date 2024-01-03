import os
import argparse
import time
from tqdm import tqdm
import numpy as np
import logging
from collections import defaultdict, deque
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip


def apply_circular_blur(frame, x_min, y_min, x_max, y_max):
    """
    Applies a circular blur to a region in a frame.

    Parameters:
    frame (np.array): The frame to apply the blur to.
    x_min, y_min, x_max, y_max (int): The coordinates of the region to blur.

    Returns:
    np.array: The frame with the blurred region.
    """
    # Calculate the center of the region to blur
    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))

    # Calculate the radius of the region to blur
    radius = int(max((x_max - x_min) // 2, (y_max - y_min) // 2))

    # Create a mask of the same size as the frame
    mask = np.zeros(frame.shape[:2], dtype="uint8")

    # Draw a filled circle (i.e., a disk) on the mask at the center of the region to blur
    cv2.circle(mask, center, radius, 255, -1)

    # Apply a Gaussian blur to the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)

    # Use the mask to replace the region to blur in the original frame with the corresponding region in the blurred frame
    frame = np.where(mask[:, :, None] == 255, blurred_frame, frame)

    return frame


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

#  Function to draw bounding box and keypoints on the frame


def draw_bounding_box_and_keypoints(frame, keypoints, person_id):
    """
    Draws a bounding box and keypoints on a frame.

    Parameters:
    frame (np.array): The frame to draw on.
    keypoints (list): The keypoints to draw.
    person_id (int): The identifier of the person.

    Returns:
    np.array: The frame with the bounding box and keypoints drawn on it.
    """
    # Convert keypoints to a numpy array
    keypoints = np.array(keypoints)

    # Scale keypoints according to frame dimensions
    keypoints = scale_keypoints(keypoints, frame.shape[1], frame.shape[0])

    # Filter out invalid keypoints
    keypoints = filter_invalid_keypoints(keypoints)

    # Only calculate and draw the bounding box if there are valid keypoints
    if keypoints.size > 0:
        # Calculate the bounding box for the face
        x_min, y_min, x_max, y_max = calculate_bounding_box(
            keypoints, frame.shape)

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw the keypoints on the frame
        for keypoint in keypoints:
            cv2.circle(frame, (int(keypoint[0]), int(
                keypoint[1])), 2, (0, 0, 255), -1)

    return frame


# Define a dictionary to store the bounding boxes of the previous frames for each person
previous_bboxes = defaultdict(lambda: deque(maxlen=10))


def process_keypoints_and_blur_faces(frame, keypoints, person_id):
    """
    Processes keypoints and applies a blur to the face region in a frame.

    Parameters:
    frame (np.array): The frame to process.
    keypoints (list): The keypoints to process.
    person_id (int): The identifier of the person.

    Returns:
    np.array: The frame with the blurred face region.
    """
    global previous_bboxes  # Access the global variable that stores previous bounding boxes

    keypoints = np.array(keypoints)  # Convert keypoints to numpy array

    # Initialize bounding box variables
    x_min, y_min, x_max, y_max = None, None, None, None

    # If there are keypoints
    if keypoints.shape[0] > 0:

        # Scale keypoints according to frame dimensions
        keypoints = scale_keypoints(keypoints, frame.shape[1], frame.shape[0])

        # Filter out invalid keypoints
        valid_keypoints = filter_invalid_keypoints(keypoints)

        margin = 50  # Define a margin to increase the size of the bounding box

        # If there are valid keypoints
        if valid_keypoints.shape[0] > 0:
            x_min, y_min, x_max, y_max = calculate_bounding_box(
                valid_keypoints, frame.shape, margin)

        # If bounding box variables have been assigned
        if None not in [x_min, y_min, x_max, y_max]:

            # Add the current bounding box to the list of previous bounding boxes for this person
            previous_bboxes[person_id].append((x_min, y_min, x_max, y_max))

            # Calculate the average bounding box coordinates for this person
            x_min, y_min, x_max, y_max = np.mean(
                previous_bboxes[person_id], axis=0).astype(int)

            # Apply circular blur to the region defined by the bounding box
            frame = apply_circular_blur(frame, x_min, y_min, x_max, y_max)

    return frame


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


# Function to extract the dimensions of the video frames, frame rate and total number of frames
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


def process_video(video_path, keypoints_dir, output_path, frames_dir, show_progress):
    """
    Process the video and apply blur to faces.

    Parameters:
    video_path (str): Path to the input video.
    keypoints_dir (str): Directory containing keypoints.
    output_path (str): Path to the output video.

    Returns:
    None
    """
    # Extract the video name from the video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Get the dimensions of the video frames, frame rate and total number of frames
    frame_width, frame_height, fps, total_frames = get_video_properties(
        video_path)

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

    # Process each frame in the video
    while cap.isOpened():
        # Update the progress bar if show_progress is True
        if show_progress:
            progress_bar.update(1)

        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # If no frame is returned, break the loop

        # Construct the keypoints file name and path
        keypoints_file_name = f"{video_name}_{frame_number}.txt"
        keypoints_file_path = os.path.join(keypoints_dir, keypoints_file_name)

        # If the keypoints file exists
        if os.path.exists(keypoints_file_path):
            # Open the keypoints file
            with open(keypoints_file_path, 'r') as file:
                # Read all lines from the file
                lines = file.readlines()
                # Process each line in the file
                for line in lines:
                    # Convert the line into a list of floats
                    data = [float(x) for x in line.split()]
                    # Extract the face keypoints from the data
                    face_keypoints = [(data[i], data[i+1], data[i+2])
                                      for i in range(5, 5+5*3, 3)]
                    # If there are no facial keypoints
                    if all(keypoint[:2] == (0, 0) for keypoint in face_keypoints):
                        # Calculate the time in the video
                        time_in_video = calculate_time(frame_number, fps)
                        # Log the frame number, the time in the video, and the lack of keypoints
                        logging.warning(
                            f"No facial keypoints found for frame {frame_number} at time {time_in_video} seconds. Keypoints: {face_keypoints}")
                        # Define the output file name
                        output_file_name = f"frame_{frame_number}.jpg"
                        # Define the output file path
                        output_file_path = os.path.join(
                            frames_dir, output_file_name)
                        # Save the frame to the output file
                        cv2.imwrite(output_file_path, frame)
                    # Process the keypoints and blur the face in the frame
                    frame = process_keypoints_and_blur_faces(
                        frame, face_keypoints, data[len(data)-1])
                    # frame = draw_bounding_box_and_keypoints(
                    #     frame, face_keypoints, data[len(data)-1])
        else:  # If the keypoints file does not exist
            # Log a warning
            logging.warning(
                f"Keypoints file not found for frame {frame_number}: {keypoints_file_name}")

        # If the frame is valid
        if frame is not None:
            # Write the frame to the output video
            out.write(frame)
        else:  # If the frame is invalid
            # Print an error message
            print("Invalid frame")
        frame_number += 1

    # Close the progress bar if show_progress is True
    if show_progress:
        progress_bar.close()

    # Close the video file, video writer and destroy all windows.
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def combine_audio_video(audio_path, video_path, output_path):
    """
    Combines the audio from one file with the video from another file.

    Parameters:
    audio_path (str): Path to the audio file.
    video_path (str): Path to the video file.
    output_path (str): Path to the output file.

    Returns:
    None
    """
    try:
        # Load the video file
        video_clip = VideoFileClip(video_path)
        # Load the audio file
        audio_clip = AudioFileClip(audio_path)
        # Set the audio of the video clip to the audio clip
        video_clip_with_audio = video_clip.set_audio(audio_clip)
        # Write the video clip with audio to the output file
        video_clip_with_audio.write_videofile(output_path, codec="libx264")
        # Log a message indicating that the combined video and audio was saved
        logging.info(f"Combined video and audio saved to {output_path}")
    except Exception as e:
        # Log an error message if an exception occurred
        logging.error(f"Error combining audio and video: {e}")


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


def setup_logging(log_file=None):
    """
    Set up logging.

    Parameters:
    log_file (str): Path to the log file.

    Returns:
    None
    """
    # Set up logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(filename=log_file,
                            level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)


def main():
    """
    Main function.

    Processes the video and applies blur to faces.

    Parameters:
    video (str): Path to the input video.
    keypoints (str): Directory containing keypoints.
    audio (str): Path to the audio file.
    output (str): Path to the output video.
    log (str): Path to the log file.
    show_progress (bool): Whether to show a progress bar.

    Returns:
    None
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Process video and apply blur to faces.')
    parser.add_argument('--video', required=True,
                        help='Path to the input video.')
    parser.add_argument('--keypoints', required=True,
                        help='Directory containing keypoints.')
    parser.add_argument('--output', required=True,
                        help='Path to the output video.')
    parser.add_argument('--log', action='store_true', help='Enable logging.')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save frames without keypoints.')
    parser.add_argument("--show_progress",
                        action="store_true", help="Show progress bar")

    # Parse the arguments
    args = parser.parse_args()

    # Extract the video file name (without extension) from the video path
    video_file_name = os.path.splitext(os.path.basename(args.video))[0]

    # Current time stamp
    time_stamp = int(time.time())

    # Current run directory
    current_run = f"runs/{video_file_name}_{time_stamp}"

    # Create the output directories if they don't exist
    if args.save_frames:
        missing_frames_dir = make_directory(
            f"{current_run}/missing_frames")

    if args.log:
        log_files_dir = make_directory(
            f"{current_run}/logs")
        setup_logging(f"{log_files_dir}/{video_file_name}_{time_stamp}.log")

    # Process the video
    process_video(args.video, args.keypoints, args.output,
                  f"{missing_frames_dir}", args.show_progress)
    # Combine the processed video with the audio
    # combine_audio_video(args.audio, args.output, args.output)


if __name__ == '__main__':
    """
    Entry point.

    Returns:
    None
    """

    main()
