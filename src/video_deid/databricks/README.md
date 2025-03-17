# Databricks Integration for Video-DeID

This module provides integration between the Video-DeID package and Databricks, allowing you to run video de-identification workflows in Databricks environments.

## Features

- Extract keypoints from videos stored in Databricks volumes
- De-identify videos by blurring faces based on keypoints
- Complete de-identification with skeleton visualization
- Batch processing of multiple videos
- Integration with Spark DataFrames for processing results
- Command-line interface for running in Databricks notebooks

## Getting Started

### Installation

1. Install the `video-deid` package in your Databricks environment:

```python
%pip install git+https://github.com/kbjohnson-penn/video-deid.git
```

2. Upload the YOLO pose model to your Databricks workspace:

```python
# Example: Upload model to DBFS
dbutils.fs.cp("file:/local/path/to/yolo11x-pose.pt", "dbfs:/FileStore/models/yolo11x-pose.pt")
```

### Usage

There are two main ways to use the Databricks integration:

#### 1. Using the CLI Interface

The CLI interface is designed to be used in Databricks notebooks and provides a simple way to run video de-identification operations.

```python
from video_deid.databricks import main_databricks

# Configure command-line arguments
import sys
sys.argv = [
    "video_deid",
    "--operation_type", "extract",  # or "deid" or "batch"
    "--video_volume_path", "/Volumes/catalog/schema/videos/example.mp4",
    "--keypoints_volume_path", "/Volumes/catalog/schema/outputs/keypoints.csv",
    "--model_volume_path", "/Workspace/Repos/user/models/yolo11x-pose.pt",
    "--log"
]

# Run the operation
result = main_databricks(spark, dbutils)
print(f"Operation result: {result}")
```

#### 2. Using the Direct API

For more control, you can use the direct API functions:

```python
from video_deid.databricks import run_video_deid_extraction, run_video_deid_blur

# Extract keypoints
keypoints_path = run_video_deid_extraction(
    spark,
    dbutils,
    "/Workspace/Repos/user/models/yolo11x-pose.pt",  # Model path
    "/Volumes/catalog/schema/videos/example.mp4",     # Video path
    "/Volumes/catalog/schema/outputs/keypoints.csv"   # Output path
)

# De-identify video
output_path = run_video_deid_blur(
    spark,
    dbutils,
    "/Volumes/catalog/schema/videos/example.mp4",     # Video path
    "/Volumes/catalog/schema/outputs/keypoints.csv",  # Keypoints path
    "/Volumes/catalog/schema/outputs/deid_video.mp4", # Output path
    complete_deid=False  # Set to True for complete de-identification
)
```

## Operations

### 1. Extract Keypoints

Extracts keypoints from a video using the YOLO pose model:

```python
sys.argv = [
    "video_deid",
    "--operation_type", "extract",
    "--video_volume_path", "/path/to/video.mp4",
    "--keypoints_volume_path", "/path/to/output/keypoints.csv",
    "--model_volume_path", "/path/to/model.pt",
    "--log"
]
result = main_databricks(spark, dbutils)
```

### 2. De-identify Video

De-identifies a video by blurring faces:

```python
sys.argv = [
    "video_deid",
    "--operation_type", "deid",
    "--video_volume_path", "/path/to/video.mp4",
    "--keypoints_volume_path", "/path/to/keypoints.csv",
    "--output_volume_dir", "/path/to/output/directory",
    "--model_volume_path", "/path/to/model.pt",
    "--log"
]
result = main_databricks(spark, dbutils)
```

For complete de-identification (blur entire video and show skeleton), add the `--complete_deid` flag:

```python
sys.argv = [
    "video_deid",
    "--operation_type", "deid",
    "--video_volume_path", "/path/to/video.mp4",
    "--keypoints_volume_path", "/path/to/keypoints.csv",
    "--output_volume_dir", "/path/to/output/directory",
    "--model_volume_path", "/path/to/model.pt",
    "--complete_deid",
    "--log"
]
result = main_databricks(spark, dbutils)
```

### 3. Batch Processing

Process multiple videos in a directory:

```python
sys.argv = [
    "video_deid",
    "--operation_type", "batch",
    "--videos_volume_dir", "/path/to/videos/directory",
    "--output_volume_dir", "/path/to/output/directory",
    "--model_volume_path", "/path/to/model.pt",
    "--file_pattern", "*.mp4",
    "--log"
]
result = main_databricks(spark, dbutils)

# View the batch processing report
if result["status"] == "success":
    report_df = spark.read.csv(result["report_path"], header=True, inferSchema=True)
    display(report_df)
```

## Example Notebook

See the `example_notebook.py` file for a complete example of how to use the Databricks integration in a notebook.

## Utilities

The module also provides utility functions for working with Databricks volumes:

- `copy_from_volume_to_local`: Copy a file from a Databricks volume to local storage
- `write_dataframe_to_volume`: Write a Spark DataFrame to a Databricks volume
- `write_single_csv_file`: Write a Spark DataFrame as a single CSV file

## Limitations

- The module requires the YOLO pose model to be accessible in the Databricks environment
- Processing is currently done locally within the Databricks executor, not distributed across the cluster
- Large videos may require significant memory resources
