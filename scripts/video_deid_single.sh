#!/bin/bash
#SBATCH --job-name=video_deid                                    # Job name
#SBATCH --output=video_deid_%A_%a.out                            # Output file name (%A is the job array ID, %a is the task index)
#SBATCH --error=video_deid_%A_%a.err                             # Error file name
#SBATCH --mail-user=your@email.com                                # Email notifications
#SBATCH --mail-type=END,FAIL                                     # Notify on job completion or failure
#SBATCH --ntasks=1                                               # Number of tasks per array job
#SBATCH --cpus-per-task=64                                       # Number of CPU cores
#SBATCH --mem-per-cpu=2G                                         # Memory per core
#SBATCH --time=1-10:00:00                                        # Maximum runtime

# Load the required modules
module load python/3.11

# Activate the virtual environment
source /path/to/venv/bin/activate

# Arguments paths
PYTHON_SCRIPT_PATH="/path/to/video_deid.py"
VIDEO="/path/to/video.mp4"
KEYPOINTS_CSV_PATH="/path/to/keypoints.csv"
OUTPUT_VIDEO="/path/to/output.mp4"
OPERATION_TYPE="deid"

# python "$PYTHON_SCRIPT_PATH" --operation_type "$OPERATION_TYPE" --video "$VIDEO" --keypoints_csv "$KEYPOINTS_CSV" --output "$OUTPUT_VIDEO"  --complete_deid --log --progress --notemp
python "$PYTHON_SCRIPT_PATH" --operation_type "$OPERATION_TYPE" --video "$VIDEO" --keypoints_csv "$KEYPOINTS_CSV" --output "$OUTPUT_VIDEO"  --log --progress