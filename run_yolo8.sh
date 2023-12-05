#!/bin/bash

# Directory containing videos
VIDEO_DIR="/path/to/video/directory"

# Directory where pose-detection.py is located
SCRIPT_DIR="/path/to/script/directory"

# Directory to store keypoints
KEYPOINTS_DIR="$VIDEO_DIR/keypoints"
mkdir -p "$KEYPOINTS_DIR"

# Iterate over all .mp4 files in the directory
for video in "$VIDEO_DIR"/*.mp4; do
    # Skip if the directory is empty
    [ -e "$video" ] || continue

    # Replace spaces in filename with underscores
    renamed_video=$(echo "$video" | tr ' ' '_')
    if [ "$video" != "$renamed_video" ]; then
        mv "$video" "$renamed_video"
        video=$renamed_video
    fi

    # Extract filename without extension for keypoints file
    filename=$(basename -- "$renamed_video")
    filename="${filename%.*}"

    # Run pose-detection script on the video
    python "$SCRIPT_DIR/pose-detection.py" "$renamed_video" "$KEYPOINTS_DIR/${filename}_keypoints.txt"
done
