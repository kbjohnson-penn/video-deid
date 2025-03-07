"""
Command-line interface for the video-deid package
"""
import os
import argparse
import logging
import tempfile
import shutil
from pathlib import Path

from .utils import (
    create_run_directory_and_paths,
    setup_logging,
    load_dataframe_from_csv,
    interpolate_and_sort_df
)
from .audio import combine_audio_video
from .blur import process_video
from .deid import blur_video, process_blurred_video
from .keypoints import extract_keypoints_and_save
from .config import DEFAULT_YOLO_MODEL


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='De-identify or extract keypoints from a video.')

    # Required arguments
    parser.add_argument('--operation_type', required=True, choices=['extract', 'deid'],
                        help='Operation to perform: "extract" keypoints or "deid" video')

    # Video input
    parser.add_argument('--video', required=True,
                        help='Path to the input video file')

    # Operation-specific arguments
    parser.add_argument('--keypoints_csv',
                        help='Path to save keypoints CSV (for extract) or path to keypoints CSV file (for deid)')
    parser.add_argument('--output',
                        help='Path to the output de-identified video (required for deid)')

    # Options
    parser.add_argument('--model', default=None,
                        help='Path to the YOLO pose model file (defaults to yolo11x-pose.pt in helpers directory)')
    parser.add_argument('--complete_deid', action='store_true',
                        help='Completely de-identify the video (blur entire video) and apply skeleton')
    parser.add_argument('--notemp', action='store_true',
                        help='Do not use temporary files, save all files in the runs directory')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging')
    parser.add_argument('--progress', action='store_true',
                        help='Show progress bar')

    args = parser.parse_args()

    # Validate operation-specific required arguments
    if args.operation_type == 'extract' and not args.keypoints_csv:
        parser.error("--keypoints_csv is required for extract operation")
    if args.operation_type == 'deid':
        if not args.keypoints_csv:
            parser.error("--keypoints_csv is required for deid operation")
        if not args.output:
            parser.error("--output is required for deid operation")

    return args


def load_keypoints(keypoints_csv):
    """Load keypoints dataframe from CSV."""
    logging.info(f'Loading keypoints dataframe from CSV: {keypoints_csv}')
    try:
        return load_dataframe_from_csv(keypoints_csv)
    except FileNotFoundError:
        logging.error(f"Keypoints CSV file not found: {keypoints_csv}")
        raise
    except Exception as e:
        logging.error(f"Error loading keypoints from CSV: {e}")
        raise


def interpolate_keypoints(keypoints_dataframe, output_path):
    """Interpolate and sort the keypoints dataframe and save to CSV."""
    logging.info('Interpolating and sorting dataframe...')
    interpolated_keypoints_df = interpolate_and_sort_df(keypoints_dataframe)
    interpolated_keypoints_df.to_csv(output_path, index=False)
    logging.info(f'Keypoints interpolated and saved to: {output_path}')
    return interpolated_keypoints_df


def process_video_with_audio(args, keypoints_dataframe, interpolated_keypoints_df, kalman_filtered_csv_path, output_video_path):
    """Process the video and combine audio."""
    try:
        logging.info(f'Processing video: {args.video}')
        process_video(args.video, keypoints_dataframe, interpolated_keypoints_df,
                      kalman_filtered_csv_path, output_video_path, args.progress)

        logging.info('Combining audio from original with processed video')
        combine_audio_video(args.video, output_video_path, args.output)
        logging.info(f'Final output saved to: {args.output}')
    except Exception as e:
        logging.error(f"An error occurred during video processing: {e}")
        raise


def deidentify_video(args, keypoints_dataframe, output_video_path):
    """Completely de-identify the video by blurring all frames."""
    try:
        logging.info(
            f"Starting complete de-identification for video: {args.video}")

        # Blur the entire video
        blur_video(args.video, output_video_path, args.progress)
        logging.info("Video blurring completed")

        # Create a temporary file for the keypoints overlay
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_keypoints_file:
            temp_keypoints_path = temp_keypoints_file.name
        
        try:
            if keypoints_dataframe is not None:
                logging.info("Overlaying keypoints on blurred video...")
                process_blurred_video(output_video_path,
                                      keypoints_dataframe, temp_keypoints_path, args.progress)
                logging.info("Keypoints overlay completed")
                
                # Combine audio from original with the keypoints video
                logging.info("Combining audio from original with processed video")
                combine_audio_video(args.video, temp_keypoints_path, args.output)
            else:
                logging.info("No keypoints provided, processing fully blurred video")
                # Combine audio from original with the blurred video
                logging.info("Combining audio from original with blurred video")
                combine_audio_video(args.video, output_video_path, args.output)
            
            logging.info(f"Final output saved to: {args.output}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_keypoints_path):
                os.unlink(temp_keypoints_path)
                logging.debug(f"Removed temporary keypoints video: {temp_keypoints_path}")

        logging.info(f"De-identification complete for: {args.video}")
    except Exception as e:
        logging.error(f"An error occurred during de-identification: {e}")
        raise


def get_model_path(custom_path=None):
    """
    Get the path to the YOLO model.

    If custom_path is provided, uses that path.
    If no custom_path is provided, uses the default model name.
    YOLO will automatically download the model if it doesn't exist.

    Parameters:
    - custom_path: Optional custom path to the YOLO model

    Returns:
    - Path to the model (either custom or default)
    """
    # If custom path is provided, use it
    if custom_path:
        return str(Path(custom_path))

    # No custom path, use default model name from config
    # YOLO will automatically download this if it doesn't exist
    return DEFAULT_YOLO_MODEL


def cleanup_temp_files(temp_file_path):
    """Clean up temporary files."""
    try:
        temp_path = Path(temp_file_path)
        if temp_path.exists():
            temp_path.unlink()
            logging.debug(f"Temporary file removed: {temp_file_path}")
    except Exception as e:
        logging.warning(
            f"Failed to cleanup temporary file {temp_file_path}: {e}")


def main():
    """Main function. Parses the arguments and processes the video."""
    temp_files = []

    try:
        args = parse_arguments()

        # Set up paths and logging
        paths = create_run_directory_and_paths(args.video)
        setup_logging(paths['log_file'] if args.log else None)
        logging.info(f"Arguments: {args}")

        # Get model path
        try:
            yolo_model = get_model_path(args.model)
            logging.info(f"Using YOLO model: {yolo_model}")
        except FileNotFoundError as e:
            logging.error(str(e))
            return

        if args.operation_type == 'extract':
            logging.info(f'Extracting keypoints from video: {args.video}')
            extract_keypoints_and_save(
                yolo_model, args.video, args.keypoints_csv)
            logging.info(
                f'Keypoints extracted and saved to: {args.keypoints_csv}')
            return

        elif args.operation_type == 'deid':
            # Load and interpolate keypoints
            try:
                keypoints_dataframe = load_keypoints(args.keypoints_csv)
                interpolated_keypoints_df = interpolate_keypoints(
                    keypoints_dataframe, paths['interpolated_csv'])
            except Exception as e:
                logging.error(f"Failed to process keypoints: {e}")
                return

            # Determine output paths based on --notemp flag
            if args.notemp:
                run_dir = Path(paths['run_directory'])
                output_video_path = run_dir / Path(args.output).name
                kalman_filtered_csv_path = run_dir / 'kalman_filtered_keypoints.csv'
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    output_video_path = temp_file.name
                    temp_files.append(output_video_path)
                kalman_filtered_csv_path = paths['kalman_filtered_csv']

            if args.complete_deid:
                # Completely de-identify the video
                deidentify_video(args, keypoints_dataframe, output_video_path)
            else:
                # Process the video and combine audio
                process_video_with_audio(args, keypoints_dataframe,
                                         interpolated_keypoints_df, kalman_filtered_csv_path, output_video_path)

        else:
            # This should never happen due to choices in argparse
            logging.error(f"Invalid operation type: {args.operation_type}")
            return

        logging.info('Video processing completed successfully.')

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            cleanup_temp_files(temp_file)


if __name__ == '__main__':
    main()
