import os
import argparse
import time
import logging
import tempfile
from helpers.utils import setup_logging, make_directory, interpolate_and_sort_df
from helpers.video_blur import process_video
from helpers.audio import combine_audio_video
from helpers.generate_keypoints_dataframe import create_keypoints_dataframe_from_labels


logging.getLogger().setLevel(logging.ERROR)


def main():
    """"
    Main function. Parses the arguments and processes the video.

    Returns:
    - None
    """

    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Process video and apply blur to faces.')
    parser.add_argument('--video', required=True,
                        help='Path to the input video.')
    parser.add_argument('--keypoints', required=True,
                        help='Path to the keypoints labels.')
    parser.add_argument('--output', required=True,
                        help='Path to the output video.')
    parser.add_argument('--log', action='store_true', help='Enable logging.')
    parser.add_argument("--progress",
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
    if args.log:
        log_files_dir = make_directory(f"{current_run}/logs")
        setup_logging(
            f"{log_files_dir}/{video_file_name}_{time_stamp}.log")

    # Create the current directory if it doesn't exist
    make_directory(current_run)
    logging.info('Created current run directory.')

    # Create the keypoints dataframe
    logging.info('Creating keypoints dataframe.')
    keypoints_dataframe = create_keypoints_dataframe_from_labels(
        args.video, args.keypoints)

    # Save the keypoints dataframe to a CSV file
    logging.info('Saving keypoints dataframe to CSV.')
    keypoints_dataframe.to_csv(
        f"{current_run}/{video_file_name}_dataframe.csv", index=False)

    # Interpolate and sort the dataframe
    logging.info('Interpolating and sorting dataframe.')
    interpolated_keypoints_df = interpolate_and_sort_df(keypoints_dataframe)

    # Save the interpolated dataframe to a CSV file
    logging.info('Finished interpolating and sorting dataframe.')
    interpolated_keypoints_df.to_csv(
        f"{current_run}/{video_file_name}_interpolated.csv", index=False)

    # Process the video
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    logging.info('Processing video.')
    process_video(args.video, keypoints_dataframe,
                  interpolated_keypoints_df, temp_file.name, args.progress)
    # process_video(args.video, keypoints_dataframe, interpolated_keypoints_df, args.output,
    #              args.progress)

    logging.info('Comining audio and video.')

    # Combine the audio and video
    combine_audio_video(args.video, temp_file.name,  args.output)

    # Delete the temporary file
    os.remove(temp_file.name)

    logging.info('Finished processing video.')


if __name__ == '__main__':
    """
    Entry point.

    Returns:
    - None
    """

    main()
