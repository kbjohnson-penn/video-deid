"""
Databricks integration for keypoint extraction
"""
import logging
import tempfile
import os
import csv
from pathlib import Path

from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from ..keypoints.extraction import extract_keypoints_and_save
from .utils import copy_from_volume_to_local, write_single_csv_file


def extract_keypoints_in_databricks(model_volume_path, video_volume_path, output_volume_path, spark, dbutils):
    """
    Extract keypoints from a video in Databricks environment

    Parameters:
    - model_volume_path: Path to the YOLO model in Databricks volume/workspace
    - video_volume_path: Path to the video in Databricks volume
    - output_volume_path: Path to save the output CSV in Databricks volume
    - spark: SparkSession object
    - dbutils: Databricks utilities object

    Returns:
    - None
    """
    temp_dir = tempfile.mkdtemp()
    logging.info(f"Created temporary directory: {temp_dir}")

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_volume_path)
        dbutils.fs.mkdirs(output_dir)

        # Copy files from volumes to local temp directory
        local_model_path = os.path.join(temp_dir, "yolo11x-pose.pt")
        local_video_path = os.path.join(temp_dir, "video.mp4")
        local_csv_path = os.path.join(temp_dir, "keypoints.csv")

        # Copy files
        copy_from_volume_to_local(
            model_volume_path, local_model_path, spark, dbutils)
        copy_from_volume_to_local(
            video_volume_path, local_video_path, spark, dbutils)

        # Extract keypoints using existing function
        extract_keypoints_and_save(
            local_model_path, local_video_path, local_csv_path)

        # Read CSV with Python's csv module
        df_rows = []
        with open(local_csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            for row in reader:
                df_rows.append(row)

        # Create DataFrame from rows with proper schema
        schema_fields = []
        for header in headers:
            # Use StringType for person_id and DoubleType for all numeric fields
            if header == 'person_id':
                schema_fields.append(StructField(header, StringType(), True))
            else:
                schema_fields.append(StructField(header, DoubleType(), True))

        schema = StructType(schema_fields)

        # Convert row data with consistent typing
        rows = []
        for row in df_rows:
            row_dict = {}
            for i, val in enumerate(row):
                header = headers[i]
                if header == 'person_id':
                    row_dict[header] = val
                else:
                    try:
                        # Convert all numeric values to float (Double)
                        row_dict[header] = float(val) if val else None
                    except (ValueError, TypeError):
                        # Handle any conversion errors
                        row_dict[header] = None
            rows.append(Row(**row_dict))

        keypoints_df = spark.createDataFrame(rows, schema=schema)

        # Write DataFrame to output
        write_single_csv_file(keypoints_df, output_volume_path, dbutils)
        logging.info(
            f"Keypoint extraction complete. Results saved to: {output_volume_path}")

    except Exception as e:
        logging.error(f"Error in Databricks processing: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    finally:
        import shutil
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory")
