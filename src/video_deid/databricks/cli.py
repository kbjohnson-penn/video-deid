"""
Command-line interface for running video-deid in Databricks
This module provides a simplified CLI for executing video-deid operations in Databricks.
It acts as an adapter between the standard CLI and Databricks runtime.
"""
import argparse
import logging
import os
from pathlib import Path

from .runner import run_video_deid_extraction, run_video_deid_blur
from .utils import write_dataframe_to_volume
from ..config import DEFAULT_YOLO_MODEL


def parse_databricks_arguments():
    """
    Parse command line arguments for Databricks operations.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='De-identify or extract keypoints from a video in Databricks.')

    # Required arguments
    parser.add_argument('--operation_type', required=True, choices=['extract', 'deid', 'batch'],
                        help='Operation to perform: "extract" keypoints, "deid" video, or "batch" process multiple videos')

    # Common arguments
    parser.add_argument('--video_volume_path',
                        help='Path to the input video in Databricks volume')
    parser.add_argument('--output_volume_dir',
                        help='Directory in Databricks volume to save outputs')

    # Operation-specific arguments
    parser.add_argument('--keypoints_volume_path',
                        help='Path to read/write keypoints CSV in Databricks volume')
    parser.add_argument('--model_volume_path',
                        help=f'Path to YOLO model in Databricks volume (defaults to {DEFAULT_YOLO_MODEL})')

    # Batch processing arguments
    parser.add_argument('--videos_volume_dir',
                        help='Directory in Databricks volume containing videos for batch processing')
    parser.add_argument('--file_pattern', default='*.mp4',
                        help='File pattern for batch processing (default: *.mp4)')

    # Options
    parser.add_argument('--complete_deid', action='store_true',
                        help='Completely de-identify the video (blur entire video) and apply skeleton')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging')

    args = parser.parse_args()

    # Validate arguments based on operation type
    if args.operation_type == 'extract':
        if not args.video_volume_path:
            parser.error(
                "--video_volume_path is required for extract operation")
        if not args.keypoints_volume_path:
            parser.error(
                "--keypoints_volume_path is required for extract operation")

    elif args.operation_type == 'deid':
        if not args.video_volume_path:
            parser.error("--video_volume_path is required for deid operation")
        if not args.keypoints_volume_path:
            parser.error(
                "--keypoints_volume_path is required for deid operation")
        if not args.output_volume_dir:
            parser.error("--output_volume_dir is required for deid operation")

    elif args.operation_type == 'batch':
        if not args.videos_volume_dir:
            parser.error("--videos_volume_dir is required for batch operation")
        if not args.output_volume_dir:
            parser.error("--output_volume_dir is required for batch operation")

    return args


def setup_databricks_logging(enable_logging=False):
    """
    Set up logging for Databricks operations.

    Args:
        enable_logging (bool): Whether to enable detailed logging
    """
    log_level = logging.INFO if enable_logging else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def extract_keypoints_operation(args, spark, dbutils):
    """
    Execute keypoint extraction operation in Databricks.

    Args:
        args (argparse.Namespace): Command line arguments
        spark: SparkSession object
        dbutils: Databricks utilities object

    Returns:
        str: Path to the output keypoints CSV
    """
    model_path = args.model_volume_path or DEFAULT_YOLO_MODEL
    logging.info(f"Extracting keypoints using model: {model_path}")

    output_path = run_video_deid_extraction(
        spark, dbutils,
        model_path,
        args.video_volume_path,
        args.keypoints_volume_path
    )

    logging.info(f"Keypoints extracted to: {output_path}")
    return output_path


def deid_operation(args, spark, dbutils):
    """
    Execute video de-identification operation in Databricks.

    Args:
        args (argparse.Namespace): Command line arguments
        spark: SparkSession object
        dbutils: Databricks utilities object

    Returns:
        str: Path to the output video
    """
    video_name = os.path.basename(args.video_volume_path)
    output_name = f"{os.path.splitext(video_name)[0]}_deid.mp4"
    output_path = os.path.join(args.output_volume_dir, output_name)

    logging.info(f"De-identifying video: {args.video_volume_path}")
    logging.info(f"Using keypoints from: {args.keypoints_volume_path}")
    logging.info(f"Output will be saved to: {output_path}")

    output_path = run_video_deid_blur(
        spark, dbutils,
        args.video_volume_path,
        args.keypoints_volume_path,
        output_path,
        args.complete_deid
    )

    logging.info(f"De-identified video saved to: {output_path}")
    return output_path


def batch_process(args, spark, dbutils):
    """
    Execute batch processing of multiple videos in Databricks.

    Args:
        args (argparse.Namespace): Command line arguments
        spark: SparkSession object
        dbutils: Databricks utilities object

    Returns:
        list: Paths to the output videos
    """
    # List all videos in the specified directory
    logging.info(f"Scanning for videos in: {args.videos_volume_dir}")
    video_pattern = f"{args.videos_volume_dir}/{args.file_pattern}"

    # Use Spark to list files with the pattern
    videos_df = spark.read.format("binaryFile").option(
        "pathGlobFilter", args.file_pattern).load(args.videos_volume_dir)
    video_paths = [row.path for row in videos_df.select("path").collect()]

    logging.info(f"Found {len(video_paths)} videos to process")

    # Create a report dataframe to track processing
    from pyspark.sql.types import StructType, StructField, StringType, BooleanType
    from pyspark.sql import Row

    schema = StructType([
        StructField("video_path", StringType(), False),
        StructField("keypoints_path", StringType(), True),
        StructField("output_path", StringType(), True),
        StructField("extraction_success", BooleanType(), False),
        StructField("deid_success", BooleanType(), False),
        StructField("error", StringType(), True)
    ])

    results = []
    model_path = args.model_volume_path or DEFAULT_YOLO_MODEL

    # Process each video
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]

        keypoints_path = f"{args.output_volume_dir}/{base_name}_keypoints.csv"
        output_path = f"{args.output_volume_dir}/{base_name}_deid.mp4"

        result = {
            "video_path": video_path,
            "keypoints_path": keypoints_path,
            "output_path": output_path,
            "extraction_success": False,
            "deid_success": False,
            "error": None
        }

        try:
            # Extract keypoints
            logging.info(f"Processing video: {video_path}")
            logging.info(f"Extracting keypoints to: {keypoints_path}")

            run_video_deid_extraction(
                spark, dbutils,
                model_path,
                video_path,
                keypoints_path
            )
            result["extraction_success"] = True

            # De-identify video
            logging.info(f"De-identifying video to: {output_path}")
            run_video_deid_blur(
                spark, dbutils,
                video_path,
                keypoints_path,
                output_path,
                args.complete_deid
            )
            result["deid_success"] = True

        except Exception as e:
            logging.error(f"Error processing {video_path}: {str(e)}")
            result["error"] = str(e)

        results.append(Row(**result))

    # Create results dataframe and save to output directory
    results_df = spark.createDataFrame(results, schema=schema)
    report_path = f"{args.output_volume_dir}/batch_processing_report.csv"
    write_dataframe_to_volume(results_df, report_path,
                              format="csv", spark=spark, dbutils=dbutils)

    logging.info(f"Batch processing complete. Report saved to: {report_path}")
    return report_path


def main_databricks(spark=None, dbutils=None):
    """
    Main function for Databricks CLI.

    Args:
        spark: SparkSession object
        dbutils: Databricks utilities object

    Returns:
        dict: Summary of the operation results
    """
    if spark is None or dbutils is None:
        try:
            # Try to import SparkSession from current context
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()

            # Try to get dbutils from current context
            import IPython
            dbutils = IPython.get_ipython().user_ns["dbutils"]
        except (ImportError, KeyError) as e:
            raise RuntimeError(
                "This function must be run in a Databricks environment") from e

    try:
        args = parse_databricks_arguments()
        setup_databricks_logging(args.log)

        logging.info(f"Starting Databricks operation: {args.operation_type}")

        if args.operation_type == "extract":
            output_path = extract_keypoints_operation(args, spark, dbutils)
            return {"status": "success", "operation": "extract", "output_path": output_path}

        elif args.operation_type == "deid":
            output_path = deid_operation(args, spark, dbutils)
            return {"status": "success", "operation": "deid", "output_path": output_path}

        elif args.operation_type == "batch":
            report_path = batch_process(args, spark, dbutils)
            return {"status": "success", "operation": "batch", "report_path": report_path}

    except Exception as e:
        logging.error(f"Error in Databricks operation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # This allows the module to be run directly in Databricks
    result = main_databricks()
    print(f"Operation result: {result}")
