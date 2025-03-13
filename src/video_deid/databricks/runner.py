"""
Main entry point for Databricks jobs
"""
import logging
import os
from pathlib import Path


def run_video_deid_extraction(spark, dbutils, model_path, video_path, output_path):
    """
    Run video de-identification keypoint extraction in Databricks

    Parameters:
    - spark: SparkSession
    - dbutils: Databricks utilities
    - model_path: Path to YOLO model in volume
    - video_path: Path to input video in volume
    - output_path: Path to save output CSV in volume

    Returns:
    - str: Path to the output CSV
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Import our function
    from .extract import extract_keypoints_in_databricks

    # Run the extraction
    extract_keypoints_in_databricks(
        model_path, video_path, output_path, spark, dbutils)

    return output_path


def run_video_deid_blur(spark, dbutils, video_path, keypoints_path, output_path, complete_deid=False):
    """
    Run video de-identification blurring in Databricks

    Parameters:
    - spark: SparkSession
    - dbutils: Databricks utilities
    - video_path: Path to input video in volume
    - keypoints_path: Path to keypoints CSV in volume
    - output_path: Path to save output video in volume
    - complete_deid: Whether to completely de-identify the video

    Returns:
    - str: Path to the output video
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Import utilities
    from video_deid.databricks.utils import copy_from_volume_to_local
    import tempfile
    import pandas as pd

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Copy files to local storage
        local_video = os.path.join(temp_dir, "input.mp4")
        local_keypoints = os.path.join(temp_dir, "keypoints.csv")
        local_output = os.path.join(temp_dir, "output.mp4")

        copy_from_volume_to_local(video_path, local_video, spark, dbutils)
        copy_from_volume_to_local(
            keypoints_path, local_keypoints, spark, dbutils)

        # Load keypoints
        keypoints_df = pd.read_csv(local_keypoints)

        # Import necessary functions
        if complete_deid:
            from ..deid import blur_video, process_blurred_video
            from ..audio import combine_audio_video

            # Process with complete de-identification
            temp_blur = os.path.join(temp_dir, "blurred.mp4")
            blur_video(local_video, temp_blur, show_progress=False)
            process_blurred_video(temp_blur, keypoints_df,
                                  local_output, show_progress=False)

        else:
            from ..utils import interpolate_and_sort_df
            from ..blur import process_video

            # Process with face blur only
            interpolated_df = interpolate_and_sort_df(keypoints_df)
            kalman_csv = os.path.join(temp_dir, "kalman.csv")
            process_video(local_video, keypoints_df, interpolated_df,
                          kalman_csv, local_output, show_progress=False)

        # Copy output back to volume
        output_dir = os.path.dirname(output_path)
        dbutils.fs.mkdirs(output_dir)

        # Use Spark approach for binary files
        with open(local_output, 'rb') as f:
            content = f.read()

        # Write file to volume
        dbutils.fs.put(output_path, content, overwrite=True)

        return output_path

    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
