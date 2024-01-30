# Directory containing the videos
VIDEO_DIR="/home/mopidevi/data/CSI"
OUTPUT_DIR="/home/mopidevi/data/processed_data"
LOG_DIR="/home/mopidevi/data/processed_data/log_directory"

# Name of the Anaconda environment
ENV_NAME="deid"

set -e

# Get the current directory
CURRENT_DIR=$(pwd)

# Path to the requirements file
REQUIREMENTS_FILE="$CURRENT_DIR/requirements.txt"

# Check if log directory exists, if not, create it
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Check if the Anaconda environment exists, if not, create it
if [[ $(conda env list | awk '{print $1}' | grep -w "$ENV_NAME") != "$ENV_NAME" ]]; then
    conda create -n "$ENV_NAME"
fi

# Activate the Anaconda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Install the Python packages from the requirements file
pip install -r "$REQUIREMENTS_FILE"

# Iterate over all .mp4 files in the directory and its subdirectories
find "$VIDEO_DIR" -type f -name "*.mp4" | while read -r video_file; do
    # Replace spaces in the filename with underscores
    renamed_file=$(echo "$video_file" | tr ' ' '_')

    # Rename the file if necessary
    if [ "$video_file" != "$renamed_file" ]; then
        mv "$video_file" "$renamed_file"
    fi

    # Create a log file for each video file
    LOG_FILE="$LOG_DIR/$(basename "${renamed_file%.*}").log"

    # Run the Python script on the renamed file and redirect output to the log file
    echo "Processing $renamed_file" >>git"$LOG_FILE"
    python video-deid.py --video "$renamed_file" --output "$OUTPUT_DIR/$(basename "${renamed_file%.*}").mp4" --log --progress >>"$LOG_FILE" 2>&1
done
