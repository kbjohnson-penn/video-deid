#!/bin/bash

# Directory containing the videos
VIDEO_DIR="/home/mopidevi/data/CSI/videos"
KEYPOINTS_DIR="/home/mopidevi/data/CSI/keypoints"
OUTPUT_DIR="/home/mopidevi/data/processed_videos"
LOG_DIR="/home/mopidevi/data/log_directory"

# Name of the Anaconda environment
ENV_NAME="deid"

set -e

# Get the current directory
CURRENT_DIR=$(pwd)

# Path to the requirements file
REQUIREMENTS_FILE="$CURRENT_DIR/requirements.txt"

# Check if log directory exists, if not, create it
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

# Check if output directory exists, if not, create it
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Check if the Anaconda environment exists, if not, create it
if [[ $(conda env list | awk '{print $1}' | grep -w "$ENV_NAME") != "$ENV_NAME" ]]; then
    echo "Creating Anaconda environment $ENV_NAME"
    conda create -n "$ENV_NAME" -y
fi

# Activate the Anaconda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Install the Python packages from the requirements file
echo "Installing Python packages from $REQUIREMENTS_FILE"
pip install -r "$REQUIREMENTS_FILE"

echo "Processing videos in $VIDEO_DIR"
# Iterate over all .mp4 files in the directory and its subdirectories
find "$VIDEO_DIR" -type f -name "*.mp4" | while read -r video_file; do
    # Create a log file for each video file
    LOG_FILE="$LOG_DIR/$(basename "${video_file%.*}").log"

    # Extract the subdirectory and the base name of the video file
    base_name=$(basename "${video_file%.*}")
    echo "Processing $video_file"

    # Check if keypoints directory exists, if not, skip to the next video
    if [ ! -d "$KEYPOINTS_DIR/$base_name/labels" ]; then
        echo "Key points for $video_file are missing, skipping to next video"
        continue
    fi

    # Run the Python script on the video file and redirect output to the log file
    echo "Processing $video_file" >>"$LOG_FILE"
    echo "python video-deid.py --video "$video_file" --keypoints "$KEYPOINTS_DIR/$base_name/labels" --output "$OUTPUT_DIR/$base_name.mp4" --log --progress"
    python video-deid.py --video "$video_file" --keypoints "$KEYPOINTS_DIR/$base_name/labels" --output "$OUTPUT_DIR/$base_name.mp4" --log --progress >>"$LOG_FILE" 2>&1
done