import os
import argparse
import logging
import tempfile
from helpers.utils import create_run_directory_and_paths, setup_logging, load_dataframe_from_csv, interpolate_and_sort_df
from helpers.video_blur import process_video
from helpers.audio import combine_audio_video


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process video and apply blur to faces.')
    parser.add_argument('--video', required=True,
                        help='Path to the input video.')
    parser.add_argument('--keypoints_csv', required=True,
                        help='Path to the keypoints CSV file.')
    parser.add_argument('--output', required=True,
                        help='Path to the output video.')
    parser.add_argument('--log', action='store_true', help='Enable logging.')
    parser.add_argument("--progress", action="store_true",
                        help="Show progress bar")
    return parser.parse_args()


def validate_inputs(video_path, keypoints_csv_path):
    """Validate input file paths."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file '{video_path}' does not exist.")
        if not os.path.exists(keypoints_csv_path):
            raise FileNotFoundError(
                f"Keypoints CSV file '{keypoints_csv_path}' does not exist.")
        return True
    except FileNotFoundError as e:
        logging.error(e)
        return False


def load_keypoints(keypoints_csv):
    """Load keypoints dataframe from CSV."""
    logging.info('Loading keypoints dataframe from CSV.')
    return load_dataframe_from_csv(keypoints_csv)


def interpolate_keypoints(keypoints_dataframe, output_path):
    """Interpolate and sort the keypoints dataframe and save to CSV."""
    logging.info('Interpolating and sorting dataframe.')
    interpolated_keypoints_df = interpolate_and_sort_df(keypoints_dataframe)
    interpolated_keypoints_df.to_csv(output_path, index=False)
    logging.info('Finished interpolating and sorting dataframe.')
    return interpolated_keypoints_df


def process_video_with_audio(args, keypoints_dataframe, interpolated_keypoints_df, kalman_filtered_csv_path):
    """Process the video and combine audio."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        try:
            logging.info('Processing video.')
            process_video(args.video, keypoints_dataframe, interpolated_keypoints_df,
                          kalman_filtered_csv_path, temp_file.name, args.progress)
            logging.info('Combining audio and video.')
            combine_audio_video(args.video, temp_file.name, args.output)
        except Exception as e:
            logging.error(f"An error occurred during video processing: {e}")
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
                logging.info('Temporary video file removed.')


def main():
    """Main function. Parses the arguments and processes the video."""

    args = parse_arguments()

    # Pre-validate inputs before proceeding
    if not validate_inputs(args.video, args.keypoints_csv):
        return

    # Set up paths and logging
    paths = create_run_directory_and_paths(args.video)
    setup_logging(paths['log_file'] if args.log else None)

    # Load and interpolate keypoints
    keypoints_dataframe = load_keypoints(args.keypoints_csv)
    interpolated_keypoints_df = interpolate_keypoints(
        keypoints_dataframe, paths['interpolated_csv'])

    # Process the video and combine audio
    process_video_with_audio(args, keypoints_dataframe,
                             interpolated_keypoints_df, paths['kalman_filtered_csv'])

    logging.info('Finished processing video.')


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}")
        raise  # Optionally re-raise to halt execution if needed
