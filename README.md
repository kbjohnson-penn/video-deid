# Video De-Identification

This project is a Python package designed to process videos by blurring detected faces, thereby anonymizing individuals in the footage. It leverages body and facial keypoints to locate people within each video frame and subsequently applies a blur effect. The application offers features such as extracting keypoints to CSV, interpolation for missing frames, and showing progress during processing.

This tool is particularly beneficial for those seeking to maintain privacy in video data, as it ensures the protection of individual identities.

## Installation

First, clone the repository:

```bash
git clone https://github.com/kbjohnson-penn/video-deid.git
```

Next, navigate to the project directory:

```bash
cd video-deid
```

Create a new Python virtual environment:

```bash
python3 -m venv env
```

Activate the virtual environment:

On Windows:

```bash
.\env\Scripts\activate
```

On Unix or MacOS:

```bash
source env/bin/activate
```

Finally, install the project dependencies:

```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

You'll also need to download the YOLO pose detection model. Specify the model path with the `--model` argument or allow the system to use the default model.

## Usage

The package has two main operation modes:

### 1. Extract Keypoints

First, extract keypoints from a video:

```bash
python -m video_deid.cli --operation_type extract --video <video_path> --keypoints_csv <output_csv_path> [--model <model_path>] [--log] [--progress]
```

### 2. De-identify Videos

After extracting keypoints, you can de-identify the video in two ways:

#### Face Blur Only

Blur only the faces in the video:

```bash
python -m video_deid.cli --operation_type deid --video <video_path> --keypoints_csv <keypoints_csv> --output <output_path> [--model <model_path>] [--log] [--progress]
```

#### Complete De-identification

Completely blur the entire video and overlay skeleton keypoints:

```bash
python -m video_deid.cli --operation_type deid --video <video_path> --keypoints_csv <keypoints_csv> --output <output_path> --complete_deid [--log] [--progress]
```

### Parameters

- `--operation_type` - Required, specify `extract` to extract keypoints or `deid` to de-identify video
- `--video` - Required, path to the input video file
- `--keypoints_csv` - Path to save or load keypoints CSV (required for both operations)
- `--output` - Path to the output de-identified video (required for deid operation)
- `--model` - Optional path to the YOLO pose model file
- `--complete_deid` - Enable complete de-identification (blur entire video and show skeleton)
- `--notemp` - Do not use temporary files, save all files in the runs directory
- `--log` - Enable logging
- `--progress` - Show processing progress bar

## Keypoints CSV Format

The keypoints CSV file generated during the extraction phase contains detailed pose information for each person detected in each frame. This data is used during the de-identification process to maintain body pose visualization while blurring faces.

### CSV Structure

Each row in the CSV represents a single person in a specific frame with the following columns:

1. `frame_number` - The frame number in the video (starting from 0)
2. `person_id` - A unique identifier for each tracked person (or "No detections" if no person was found)
3. `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` - Bounding box coordinates for the detected person

Then, for each of the 17 keypoints (0-16), there are three columns:
- `x_i` - X-coordinate of keypoint i
- `y_i` - Y-coordinate of keypoint i
- `c_i` - Confidence score for keypoint i (between 0 and 1)

### Keypoint Mapping

The 17 keypoints represent different body parts following the COCO keypoints format:

| Index | Body Part       |
|-------|----------------|
| 0     | Nose           |
| 1     | Left Eye       |
| 2     | Right Eye      |
| 3     | Left Ear       |
| 4     | Right Ear      |
| 5     | Left Shoulder  |
| 6     | Right Shoulder |
| 7     | Left Elbow     |
| 8     | Right Elbow    |
| 9     | Left Wrist     |
| 10    | Right Wrist    |
| 11    | Left Hip       |
| 12    | Right Hip      |
| 13    | Left Knee      |
| 14    | Right Knee     |
| 15    | Left Ankle     |
| 16    | Right Ankle    |

A keypoint with coordinates (0, 0) and low confidence indicates that the keypoint was not detected in that frame.

### Using Keypoints Data

During de-identification, the system uses:
- Keypoints 0-4 (facial keypoints) to locate and blur faces
- All keypoints to draw the skeletal visualization in complete de-identification mode

The CSV data includes tracking information, allowing the system to consistently identify the same person across different frames, which improves the stability of blurring.

## Project Structure

The project is organized into a clean package structure:

```
video_deid/
├── blur/       # Face blurring functionality
├── keypoints/  # Keypoint extraction with YOLO
├── deid/       # Complete de-identification
├── utils/      # Utility functions
├── audio.py    # Audio handling
├── cli.py      # Command-line interface
└── config.py   # Configuration parameters
```

This structure provides good separation of concerns and makes the codebase more maintainable.
