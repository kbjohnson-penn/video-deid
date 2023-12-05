#!/bin/bash

# Directory containing videos
VIDEO_DIR="/home/CSI_Kevin_Johnson"

# Directory where pose-detection.py is located
SCRIPT_DIR="/usr/src/ultralytics"

# Directory to store keypoints
KEYPOINTS_DIR="$VIDEO_DIR/keypoints"
echo "Creating keypoints directory at $KEYPOINTS_DIR"
mkdir -p "$KEYPOINTS_DIR" || { echo "Failed to create keypoints directory. Exiting."; exit 1; }

# Count the number of .mp4 files
total_files=$(ls -1 "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
current_file=0

# Iterate over all .mp4 files in the directory
for video in "$VIDEO_DIR"/*.mp4; do
    # Skip if the directory is empty
    [ -e "$video" ] || continue

    # Increment file counter
    ((current_file++))

    echo "Processing file $current_file of $total_files: $video"

    # Replace spaces in filename with underscores
    renamed_video=$(echo "$video" | tr ' ' '_')
    if [ "$video" != "$renamed_video" ]; then
        echo "Renaming $video to $renamed_video"
        mv "$video" "$renamed_video" || { echo "Failed to rename $video. Skipping to next file."; continue; }
    fi

    # Extract filename without extension for keypoints file
    filename=$(basename -- "$renamed_video")
    filename="${filename%.*}"

    # Run pose-detection script on the video
    echo "Running pose detection on $renamed_video"
    python "$SCRIPT_DIR/pose-detection.py" "$renamed_video" "$KEYPOINTS_DIR/${filename}_keypoints.txt" || { echo "Pose detection failed for $renamed_video. Skipping to next file."; continue; }
done

echo "Processing complete. Keypoints saved in $KEYPOINTS_DIR"
