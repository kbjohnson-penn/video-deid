import cv2
import numpy as np
import os
import logging
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(filename='video_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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

    # If there are keypoints
    if keypoints.shape[0] > 0:
        # Scale keypoints according to frame dimensions
        keypoints[:, 0] *= frame.shape[1]
        keypoints[:, 1] *= frame.shape[0]

        # Filter out invalid keypoints
        valid_keypoints = keypoints[(
            keypoints[:, 0] != 0) & (keypoints[:, 1] != 0)]

        margin = 50  # Define a margin to increase the size of the bounding box

        # If there are valid keypoints
        if valid_keypoints.shape[0] > 0:
            # Calculate the minimum and maximum coordinates of the bounding box
            x_min, y_min = np.min(
                valid_keypoints[:, :2], axis=0).astype(int) - margin
            x_max, y_max = np.max(
                valid_keypoints[:, :2], axis=0).astype(int) + margin
        else:  # If there's only one keypoint
            # Use the single keypoint to define the bounding box
            x_min, y_min = (keypoints[0, 0] - margin, keypoints[0, 1] - margin)
            x_max, y_max = (keypoints[0, 0] + margin, keypoints[0, 1] + margin)

        # Add the current bounding box to the list of previous bounding boxes for this person
        previous_bboxes[person_id].append((x_min, y_min, x_max, y_max))

        # Calculate the average bounding box coordinates for this person
        x_min, y_min, x_max, y_max = np.mean(
            previous_bboxes[person_id], axis=0).astype(int)

        # Ensure the bounding box coordinates are within the frame dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # Apply circular blur to the region defined by the bounding box
        frame = apply_circular_blur(frame, x_min, y_min, x_max, y_max)
    return frame


def main(video_path, keypoints_dir, output_path):
    """
    Main function to process the video and apply blur to faces.

    Parameters:
    video_path (str): Path to the input video.
    keypoints_dir (str): Directory containing keypoints.
    output_path (str): Path to the output video.
    """
    # Extract the video name from the video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    # Get the dimensions of the video frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (frame_width, frame_height))

    # Initialize frame number
    frame_number = 1

    # Process each frame in the video
    while cap.isOpened():
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
                    # Log the keypoints
                    logging.info(
                        f"Frame {frame_number} keypoints: {face_keypoints}")
                    # Process the keypoints and blur the face in the frame
                    frame = process_keypoints_and_blur_faces(
                        frame, face_keypoints, data[len(data)-1])
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

    # Close the video file, video writer and destroy all windows.
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = '/Users/mopidevi/Workspace/projects/video-deid/CSI_03.02.18_Trexler_01_TRIMMED.mp4'
    keypoints_dir = '/Users/mopidevi/Workspace/projects/video-deid/labels'
    output_path = '/Users/mopidevi/Workspace/projects/video-deid/output.mp4'

    # Call the main function
    main(video_path, keypoints_dir, output_path)
