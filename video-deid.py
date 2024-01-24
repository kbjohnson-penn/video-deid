import os
import argparse
import time
import logging
from utils import setup_logging, make_directory
from generate_keypoints_dataframe import create_keypoints_dataframe
from video_blur import process_video


logging.getLogger().setLevel(logging.ERROR)


def main():
    """"
    Main function. Parses the arguments and processes the video.

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
    parser.add_argument('--save_csv', action='store_true',
                        help='Save CSV file.')
    parser.add_argument('--log', action='store_true', help='Enable logging.')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save frames that don\'t have keypoints.')
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

    # Initialize missing_frames_dir to None
    missing_frames_dir = None

    # Create the output directories if they don't exist
    if args.log or args.save_frames:
        log_files_dir = make_directory(f"{current_run}/logs")
        if args.log:
            setup_logging(
                f"{log_files_dir}/{video_file_name}_{time_stamp}.log")
        if args.save_frames:
            missing_frames_dir = make_directory(
                f"{log_files_dir}/missing_frames")

    # Generate the keypoints dataframe
    keypoints_dataframe = create_keypoints_dataframe(
        args.video, args.keypoints)

    logging.info('Created keypoints dataframe.')

    # Save the keypoints dataframe as a CSV file
    if args.save_csv:
        make_directory(current_run)
        keypoints_dataframe.to_csv(
            f"{current_run}/keypoints_{video_file_name}_{time_stamp}.csv", index=False)
        logging.info(
            f'Saved keypoints dataframe to CSV file: {current_run}/keypoints_{video_file_name}_{time_stamp}.csv')

    # Process the video
    process_video(args.video, args.keypoints, keypoints_dataframe, args.output,
                  f"{missing_frames_dir}", args.progress)
    logging.info('Finished processing video.')


if __name__ == '__main__':
    """
    Entry point.

    Returns:
    None
    """

    main()
