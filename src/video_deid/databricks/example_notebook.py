# Databricks notebook source
# MAGIC %md
# MAGIC # Video De-identification in Databricks
# MAGIC
# MAGIC This notebook demonstrates how to use the video-deid package in Databricks to:
# MAGIC 1. Extract keypoints from videos
# MAGIC 2. De-identify videos by blurring faces
# MAGIC 3. Batch process multiple videos
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - This notebook requires the video-deid package to be installed in your Databricks environment
# MAGIC - You need a YOLO pose model (yolo11x-pose.pt) stored in your Databricks workspace
# MAGIC - Input videos must be accessible from your Databricks volumes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC First, let's import the necessary modules and define our paths

# COMMAND ----------

# Import the video-deid package
from pyspark.sql.functions import col
from video_deid.databricks import run_video_deid_extraction, run_video_deid_blur
from video_deid.databricks import main_databricks, batch_process
import sys
import os

# COMMAND ----------

# Define paths to your resources
# Path to the YOLO model
model_volume_path = "/Workspace/Repos/your_user/video-deid/models/yolo11x-pose.pt"
# Directory containing videos
videos_volume_dir = "/Volumes/your_catalog/your_schema/videos/"
output_volume_dir = "/Volumes/your_catalog/your_schema/outputs/"  # Directory for outputs

# Verify these paths exist
print(f"Model path exists: {os.path.exists(model_volume_path)}")
print(
    f"Videos directory exists: {dbutils.fs.ls(videos_volume_dir) is not None}")
print(
    f"Output directory exists: {dbutils.fs.ls(output_volume_dir) is not None}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1: Extract Keypoints from a Single Video

# COMMAND ----------

# Define the video and output paths
video_volume_path = f"{videos_volume_dir}/example_video.mp4"
keypoints_volume_path = f"{output_volume_dir}/example_video_keypoints.csv"

# Run the keypoint extraction
sys.argv = [
    "video_deid",
    "--operation_type", "extract",
    "--video_volume_path", video_volume_path,
    "--keypoints_volume_path", keypoints_volume_path,
    "--model_volume_path", model_volume_path,
    "--log"
]

# Execute using the main_databricks function
result = main_databricks(spark, dbutils)
print(f"Extraction result: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 2: De-identify a Single Video

# COMMAND ----------

# Define output path for the de-identified video
output_path = f"{output_volume_dir}/example_video_deid.mp4"

# Run the de-identification
sys.argv = [
    "video_deid",
    "--operation_type", "deid",
    "--video_volume_path", video_volume_path,
    "--keypoints_volume_path", keypoints_volume_path,
    "--output_volume_dir", output_volume_dir,
    "--model_volume_path", model_volume_path,
    "--log"
]

# Execute using the main_databricks function
result = main_databricks(spark, dbutils)
print(f"De-identification result: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 3: Batch Process Multiple Videos

# COMMAND ----------

# Define parameters for batch processing
sys.argv = [
    "video_deid",
    "--operation_type", "batch",
    "--videos_volume_dir", videos_volume_dir,
    "--output_volume_dir", output_volume_dir,
    "--model_volume_path", model_volume_path,
    "--file_pattern", "*.mp4",
    "--log"
]

# Execute using the main_databricks function
result = main_databricks(spark, dbutils)
print(f"Batch processing result: {result}")

# Display the processing report
if result["status"] == "success":
    report_df = spark.read.csv(
        result["report_path"], header=True, inferSchema=True)
    display(report_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 4: Complete De-identification (Skeleton Visualization)

# COMMAND ----------

# Define output path for the completely de-identified video
output_path = f"{output_volume_dir}/example_video_complete_deid.mp4"

# Run the complete de-identification
sys.argv = [
    "video_deid",
    "--operation_type", "deid",
    "--video_volume_path", video_volume_path,
    "--keypoints_volume_path", keypoints_volume_path,
    "--output_volume_dir", output_volume_dir,
    "--model_volume_path", model_volume_path,
    "--complete_deid",
    "--log"
]

# Execute using the main_databricks function
result = main_databricks(spark, dbutils)
print(f"Complete de-identification result: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Usage: Direct API Access
# MAGIC
# MAGIC You can also use the direct API functions if you need more control over the process

# COMMAND ----------


# Extract keypoints
custom_keypoints_path = f"{output_volume_dir}/custom_keypoints.csv"
extracted_path = run_video_deid_extraction(
    spark, dbutils,
    model_volume_path,
    video_volume_path,
    custom_keypoints_path
)
print(f"Keypoints extracted to: {extracted_path}")

# De-identify video with custom settings
custom_output_path = f"{output_volume_dir}/custom_output.mp4"
deid_path = run_video_deid_blur(
    spark, dbutils,
    video_volume_path,
    custom_keypoints_path,
    custom_output_path,
    complete_deid=True  # Use complete de-identification
)
print(f"De-identified video saved to: {deid_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing Custom Batch Processing Logic
# MAGIC
# MAGIC You can also create your own batch processing logic by combining the APIs

# COMMAND ----------

# Get a list of videos

# List videos matching a pattern
videos_df = spark.read.format("binaryFile").load(videos_volume_dir)
videos_df = videos_df.filter(col("path").endswith(".mp4"))
videos_df.select("path").show(truncate=False)

# Process specific videos with custom logic
for video_path in videos_df.select("path").limit(3).collect():
    video_path = video_path[0]
    base_name = os.path.basename(video_path).split(".")[0]

    # Create custom paths
    kp_path = f"{output_volume_dir}/{base_name}_custom_keypoints.csv"
    out_path = f"{output_volume_dir}/{base_name}_custom_output.mp4"

    # Extract keypoints
    run_video_deid_extraction(
        spark, dbutils, model_volume_path, video_path, kp_path)

    # De-identify
    run_video_deid_blur(spark, dbutils, video_path, kp_path,
                        out_path, complete_deid=False)

    print(f"Processed {base_name}")
