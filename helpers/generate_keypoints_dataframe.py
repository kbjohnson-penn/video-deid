import os
import pandas as pd
from helpers.utils import get_video_properties


def create_empty_dataframe():
    """
    Creates an empty dataframe with the required columns.

    Parameters:
    - None

    Returns:
    - pd.DataFrame: The empty dataframe.
    """

    # Define the column names
    columns = ['frame_number', 'person_id', 'x_0', 'y_0', 'c_0', 'x_1', 'y_1',
               'c_1', 'x_2', 'y_2', 'c_2', 'x_3', 'y_3', 'c_3', 'x_4', 'y_4', 'c_4']

    # Create an empty dataframe with the specified columns
    df = pd.DataFrame(columns=columns)

    return df


def append_to_dataframe(df, frame_number, person_id, keypoints):
    """
    Appends the keypoints, frame number and person identifier to a dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe to append to.
    - frame_number (int): The frame number.
    - person_id (int): The identifier of the person.
    - keypoints (list): The keypoints.

    Returns:
    - pd.DataFrame: The dataframe with the keypoints, frame number and person identifier appended.
    """

    # Create a dictionary to store the data
    data = {'frame_number': frame_number, 'person_id': person_id}

    # Add the keypoints to the dictionary
    for i, keypoint in enumerate(keypoints):
        data[f'x_{i}'] = keypoint[0]
        data[f'y_{i}'] = keypoint[1]
        data[f'c_{i}'] = keypoint[2]

    # Convert the data to a DataFrame
    data_df = pd.DataFrame([data])

    # Append the data to the dataframe using pd.concat
    df = pd.concat([df, data_df], ignore_index=True)

    return df


def create_keypoints_dataframe_from_labels(video_path, keypoints_dir):
    """
    Creates a dataframe from the keypoints.

    Parameters:
    - video_path (str): The path to the video.
    - keypoints_dir (str): The path to the directory containing the keypoints.

    Returns:
    - pd.DataFrame: The dataframe.
    """

    # Extract the video name from the video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Get the dimensions of the video frames, frame rate and total number of frames
    frame_width, frame_height, fps, total_frames = get_video_properties(
        video_path)

    # create a empty dataframe
    df = create_empty_dataframe()

    for frame_number in range(1, total_frames+1):
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

                    # append the keypoints, frame number and person id to the dataframe
                    df = append_to_dataframe(df=df, frame_number=frame_number, person_id=data[len(
                        data)-1], keypoints=face_keypoints)

    return df
