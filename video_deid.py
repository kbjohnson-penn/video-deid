import os
import argparse
import logging
import tempfile
from helpers.utils import create_run_directory_and_paths, setup_logging, load_dataframe_from_csv, interpolate_and_sort_df
from helpers.video_blur import process_video
from helpers.audio import combine_audio_video
from helpers.complete_deid import blur_video, process_blurred_video
from helpers.extract_keypoints import extract_keypoints_and_save


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='De-identify or extract keypoints from a video.')
    parser.add_argument('--operation_type', required=True,
                        help='[extract|deid] Specify if de-identifying or Extracting keypoints.')
    parser.add_argument('--video', help='Path to the input video.')
    parser.add_argument('--keypoints_csv',
                        help='Path to the keypoints CSV file.')
    parser.add_argument('--output', help='Path to the output video.')
    parser.add_argument('--complete_deid', action='store_true',
                        help='Completely de-identify the video (blur entire video) and apply skeleton.')
    parser.add_argument('--notemp', action='store_true',
                        help='Do not use temporary files, save all files in the runs directory.')
    parser.add_argument('--log', action='store_true', help='Enable logging.')
    parser.add_argument('--progress', action='store_true',
                        help='Show progress bar')
    return parser.parse_args()


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


def process_video_with_audio(args, keypoints_dataframe, interpolated_keypoints_df, kalman_filtered_csv_path, output_video_path):
    """Process the video and combine audio."""
    try:
        logging.info('Processing video.')
        process_video(args.video, keypoints_dataframe, interpolated_keypoints_df,
                      kalman_filtered_csv_path, output_video_path, args.progress)
        logging.info('Combining audio and video.')
        combine_audio_video(args.video, output_video_path, args.output)
    except Exception as e:
        logging.error(f"An error occurred during video processing: {e}")


def deidentify_video(args, keypoints_dataframe, output_video_path):
    """Completely de-identify the video by blurring all frames."""
    try:
        logging.info(
            f"Starting complete de-identification for video -- {args.video}.")
        # Blur the entire video
        blur_video(args.video, output_video_path, args.progress)
        logging.info("Completed video blurring.")

        if keypoints_dataframe is not None:
            logging.info("Processing blurred video to overlay keypoints.")
            process_blurred_video(output_video_path,
                                  keypoints_dataframe, args.output, args.progress)
        else:
            logging.info(
                f"Saving fully blurred video without keypoint annotations to {args.output}.")
            os.rename(output_video_path, args.output)

        logging.info(f"Completed processing all frames -- {args.video}.")
    except Exception as e:
        logging.error(f"An error occurred during de-identification: {e}")


def main():
    """Main function. Parses the arguments and processes the video."""

    args = parse_arguments()

    yolo_model = "/cbica/home/mopidevs/workspace/projects/pipeline/video-deid/helpers/yolo11x-pose.pt"
    
    # Set up paths and logging
    paths = create_run_directory_and_paths(args.video)
    setup_logging(paths['log_file'] if args.log else None)
    logging.info(f"Arguments: {args}")

    if args.operation_type == 'extract':
        logging.info('Extracting keypoints from video.')
        extract_keypoints_and_save(yolo_model, args.video, args.keypoints_csv)
        return

    elif args.operation_type == 'deid':
        # Load and interpolate keypoints
        keypoints_dataframe = load_keypoints(args.keypoints_csv)
        interpolated_keypoints_df = interpolate_keypoints(
            keypoints_dataframe, paths['interpolated_csv'])

        # Determine output paths based on --notemp flag
        if args.notemp:
            output_video_path = os.path.join(
                paths['run_directory'], os.path.basename(args.output))
            kalman_filtered_csv_path = os.path.join(
                paths['run_directory'], 'kalman_filtered_keypoints.csv')
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                output_video_path = temp_file.name
            kalman_filtered_csv_path = paths['kalman_filtered_csv']

        if args.complete_deid:
            # Completely de-identify the video
            deidentify_video(args, keypoints_dataframe, output_video_path)
        else:
            # Process the video and combine audio
            process_video_with_audio(args, keypoints_dataframe,
                                     interpolated_keypoints_df, kalman_filtered_csv_path, output_video_path)
            
    else:
        logging.error(f"Invalid operation type: {args.operation_type}. Please specify 'extract' or 'deid'.")
        return

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
