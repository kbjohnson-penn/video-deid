"""
Configuration settings for the video-deid package
"""
import os
from pathlib import Path

# File paths
DEFAULT_RUNS_DIR = Path("runs")

# Model settings
DEFAULT_YOLO_MODEL = "yolo11x-pose.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# Blur settings
BLUR_KERNEL_SIZE = 25
BLUR_SIGMA = 10
FACE_MARGIN = 50

# Video settings
DEFAULT_FOURCC = "mp4v"

# Keypoints settings
FACIAL_KEYPOINTS_COUNT = 5  # Number of facial keypoints in the model

# Performance settings
BATCH_SIZE = 20
LOG_FREQUENCY = 100  # Log every X frames

# Constants
KEYPOINTS_DTYPE_SPEC = {
    'frame_number': int,
    'x_0': float, 'y_0': float, 'x_1': float, 'y_1': float,
    'x_2': float, 'y_2': float, 'x_3': float, 'y_3': float,
    'x_4': float, 'y_4': float, 'x_5': float, 'y_5': float,
    'x_6': float, 'y_6': float, 'x_7': float, 'y_7': float,
    'x_8': float, 'y_8': float, 'x_9': float, 'y_9': float,
    'x_10': float, 'y_10': float, 'x_11': float, 'y_11': float,
    'x_12': float, 'y_12': float, 'x_13': float, 'y_13': float,
    'x_14': float, 'y_14': float, 'x_15': float, 'y_15': float,
    'x_16': float, 'y_16': float
}