# Directory containing the videos
VIDEO_DIR="/path/to/your/videos"
OUTPUT_DIR="/path/to/your/output"
LOG_FILE="/path/to/your/log_file.txt"

# Iterate over all .mp4 files in the directory and its subdirectories
find "$VIDEO_DIR" -type f -name "*.mp4" | while read -r video_file; do
    # Replace spaces in the filename with underscores
    renamed_file=$(echo "$video_file" | tr ' ' '_')

    # Rename the file if necessary
    if [ "$video_file" != "$renamed_file" ]; then
        mv "$video_file" "$renamed_file"
    fi

    # Run the Python script on the renamed file and redirect output to the log file
    echo "Processing $renamed_file" >>"$LOG_FILE"
    python video-deid.py --video "$renamed_file" --output "$OUTPUT_DIR/${renamed_file%.*}.mp4 --log --progress" >>"$LOG_FILE" 2>&1
done
