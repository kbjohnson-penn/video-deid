"""
Databricks volume utilities for video-deid
"""
import logging
import os
from pathlib import Path
import tempfile


def copy_from_volume_to_local(volume_path, local_path, spark, dbutils):
    """Copy file from Unity Catalog volume or workspace to local temp directory"""
    try:
        # For workspace files (model)
        if volume_path.startswith("/Workspace"):
            # Use dbutils.fs approach for workspace
            dbutils.fs.cp(f"file:{volume_path}", f"file:{local_path}")
            logging.info(
                f"Copied workspace file {volume_path} to {local_path}")
            return local_path

        # For volume files (video)
        logging.info(f"Attempting to read {volume_path}")
        # Try direct file access first
        try:
            with open(volume_path, 'rb') as src, open(local_path, 'wb') as dst:
                dst.write(src.read())
            logging.info(
                f"Direct file copy from {volume_path} to {local_path}")
            return local_path
        except Exception as e:
            logging.warning(
                f"Direct file access failed: {e}, trying Spark approach")

        # Spark approach
        df = spark.read.format("binaryFile").load(volume_path)
        if df.count() == 0:
            raise ValueError(f"No data found at {volume_path}")

        file_bytes = df.select("content").first()[0]
        with open(local_path, "wb") as f:
            f.write(file_bytes)

        logging.info(f"Copied {volume_path} to {local_path} using Spark")
        return local_path
    except Exception as e:
        logging.error(f"Failed to copy {volume_path}: {e}")
        raise


def write_dataframe_to_volume(df, volume_path, format="csv", spark=None, dbutils=None):
    """Write a dataframe to Unity Catalog volume"""
    try:
        # Ensure parent directories exist
        output_dir = os.path.dirname(volume_path)
        dbutils.fs.mkdirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

        # For CSV files
        if format.lower() == "csv":
            # Write single file CSV
            df.coalesce(1).write.format(format).mode("overwrite").option(
                "header", "true").save(volume_path + "_temp")

            # Move and rename the file to the desired path
            temp_files = dbutils.fs.ls(volume_path + "_temp")
            csv_file = next(
                (f.path for f in temp_files if f.path.endswith(".csv")), None)

            if csv_file:
                dbutils.fs.cp(csv_file, volume_path)
                dbutils.fs.rm(volume_path + "_temp", recurse=True)
                logging.info(f"Saved CSV to {volume_path}")
            else:
                logging.error(f"Failed to find CSV file in {volume_path}_temp")
        else:
            # Save as other format
            df.write.format(format).mode("overwrite").save(volume_path)
            logging.info(
                f"Saved dataframe to {volume_path} in {format} format")
    except Exception as e:
        logging.error(f"Error saving dataframe: {e}")
        raise


def write_single_csv_file(df, output_path, dbutils=None):
    """
    Write a Spark DataFrame as a single CSV file to the specified output path.

    Parameters:
    - df: Spark DataFrame to write
    - output_path: Destination path for the single CSV file
    - dbutils: Databricks utilities object
    """
    import uuid

    # Create a temporary directory with a unique name
    temp_dir = f"/tmp/spark_csv_output_{uuid.uuid4()}"

    try:
        # Write DataFrame as a single CSV file (coalesced to 1 partition)
        df.coalesce(1).write.option("header", "true").mode(
            "overwrite").csv(temp_dir)

        # Find the CSV part file in the temporary directory
        csv_files = [f.path for f in dbutils.fs.ls(
            temp_dir) if f.path.endswith(".csv")]

        if not csv_files:
            raise Exception(
                f"No CSV files found in temporary directory: {temp_dir}")

        # Get the first (and should be only) CSV file
        csv_file = csv_files[0]

        # Copy the file to the target location
        dbutils.fs.cp(csv_file, output_path)

        logging.info(
            f"Successfully wrote DataFrame as a single CSV file to: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error writing single CSV file: {e}")
        raise
    finally:
        # Clean up temporary directory
        dbutils.fs.rm(temp_dir, recurse=True)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")
