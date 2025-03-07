# Video De-identification Package

This package provides tools for de-identifying faces in videos through keypoint extraction and blurring.

## Modules

- **blur**: Face blurring functionality using Kalman filter tracking
- **deid**: Complete video de-identification with optional skeleton overlay
- **keypoints**: Keypoint extraction using YOLO pose models
- **utils**: Utility functions for file handling, keypoint processing, etc.
- **audio**: Audio extraction and combination

## Usage

### Command-line Interface

The package can be used via the command-line interface:

```bash
# Extract keypoints from a video
video-deid --operation_type extract --video input.mp4 --keypoints_csv output_keypoints.csv

# De-identify a video using keypoints
video-deid --operation_type deid --video input.mp4 --keypoints_csv keypoints.csv --output output.mp4

# Complete de-identification (blur entire video)
video-deid --operation_type deid --video input.mp4 --keypoints_csv keypoints.csv --output output.mp4 --complete_deid
```

### Programmatic Usage

```python
from video_deid import extract_keypoints_and_save, process_video

# Extract keypoints
extract_keypoints_and_save("yolo11x-pose.pt", "input.mp4", "keypoints.csv")

# Process video with keypoints
import pandas as pd
keypoints_df = pd.read_csv("keypoints.csv")
process_video("input.mp4", keypoints_df, "output.mp4")
```

## Configuration

Configuration parameters are centralized in `config.py`. You can modify these settings to customize the behavior of the package.

## Installation

```bash
pip install .
```