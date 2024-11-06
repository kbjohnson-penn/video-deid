#!/bin/bash
#SBATCH --job-name=video_deid_array                              # Job name
#SBATCH --output=video_deid_%A_%a.out                            # Output file name (%A is the job array ID, %a is the task index)
#SBATCH --error=video_deid_%A_%a.err                             # Error file name
#SBATCH --mail-user=your@email.com                               # Email notifications
#SBATCH --mail-type=END,FAIL                                     # Notify on job completion or failure
#SBATCH --ntasks=1                                               # Number of tasks per array job
#SBATCH --cpus-per-task=64                                       # Number of CPU cores
#SBATCH --mem-per-cpu=2G                                         # Memory per core
#SBATCH --time=1-10:00:00                                        # Maximum runtime
#SBATCH --array=0-23                                             # Create a job array with 19 jobs (adjust the size based on the number of videos)

# Run Using Propagate ::: sbatch --propagate=NONE scripts/video_deid_array.sh

# Load the required modules
module load python/3.11

echo "Loading Environment"
source ~/Venvs/deid/bin/activate

# Check Python location for verification
which python

# Define base directories for videos, CSVs, and output, and path for the Python script
BASE_VIDEO_DIR="/base/video/directory"
CSV_DIRECTORY="/csv/directory"
OUTPUT_DIRECTORY="/output/directory"
PYTHON_SCRIPT_PATH="/path/to/video_deid.py"
OPERATION_TYPE="deid"

# Array of video file names (without the full path)
video_files=(
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
    files.mp4
)

# Select the video file based on SLURM_ARRAY_TASK_ID
VIDEO="$BASE_VIDEO_DIR/${video_files[$SLURM_ARRAY_TASK_ID]}"

# Derive the CSV file path by replacing the directory and extension
VIDEO_BASENAME=$(basename "${VIDEO%.MP4}")
KEYPOINTS_CSV="$CSV_DIRECTORY/${VIDEO_BASENAME}.csv"
OUTPUT_VIDEO="$OUTPUT_DIRECTORY/${VIDEO_BASENAME}_deid.mp4"

# Execute the Python script with the specified parameters
echo "Running Python Script for Task ID $SLURM_ARRAY_TASK_ID"
# python "$PYTHON_SCRIPT_PATH" --operation_type "$OPERATION_TYPE" --video "$VIDEO" --keypoints_csv "$KEYPOINTS_CSV" --output "$OUTPUT_VIDEO" --log --deid --notemp
# python "$PYTHON_SCRIPT_PATH" --operation_type "$OPERATION_TYPE" --video "$VIDEO" --keypoints_csv "$KEYPOINTS_CSV" --output "$OUTPUT_VIDEO" --complete_deid --log
python "$PYTHON_SCRIPT_PATH" --operation_type "$OPERATION_TYPE" --video "$VIDEO" --keypoints_csv "$KEYPOINTS_CSV" --output "$OUTPUT_VIDEO" --log

echo "Done with Task ID $SLURM_ARRAY_TASK_ID!"
