#!/bin/bash
#SBATCH --job-name=extract_keypoints                             # Job name
#SBATCH --output=extract_keypoints_%A_%a.out                     # Output file name (%A is the job array ID, %a is the task index)
#SBATCH --error=extract_keypoints_%A_%a.err                      # Error file name
#SBATCH --mail-user=your@email.com                               # Email notifications
#SBATCH --mail-type=END,FAIL                                     # Notify on job completion or failure
#SBATCH --ntasks=1                                               # Number of tasks per job
#SBATCH --cpus-per-gpu=6                                         # Number of CPU cores per GPU
#SBATCH --gpus=a100:1                                            # Request 1 GPU
#SBATCH --mem-per-gpu=160G                                       # Memory per GPU
#SBATCH --time=06:00:00                                          # Maximum runtime
#SBATCH --array=0-4                                              # Create a job array with 19 tasks

# Load the required modules
module load python/3.11

echo "Loading Environment"
source ~/Venvs/deid/bin/activate

# Check Python location for verification
which python

# Define base directories for videos and CSVs, and path for the Python script
BASE_VIDEO_DIR="/base/video/directory"
CSV_DIRECTORY="/csv/directory"
PYTHON_SCRIPT_PATH="/path/to/video_deid.py"
OPERATION_TYPE="extract"

# Array of video file names (without the full path)
video_files=(
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

# Execute the Python script with the specified parameters
echo "Running Python Script for Task ID $SLURM_ARRAY_TASK_ID"
python "$PYTHON_SCRIPT_PATH" --operation_type "$OPERATION_TYPE" --video "$VIDEO" --keypoints_csv "$KEYPOINTS_CSV" --log

echo "Done with Task ID $SLURM_ARRAY_TASK_ID!"
