# Video De-Identification

This project is a Python script designed to process videos by blurring detected faces, thereby anonymizing individuals in the footage. It leverages body and facial keypoints to locate people within each video frame and subsequently applies a blur effect. The application offers features such as extracting keypoints to CSV, interpolation for missing frames, and showing progress during processing.

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
```

You'll also need to download the YOLO pose detection model. Place `yolo11x-pose.pt` in the `helpers` directory, or specify a custom path with the `--model` argument.

## Usage

The script has two main operation modes:

### 1. Extract Keypoints

First, extract keypoints from a video:

```bash
python video_deid.py --operation_type extract --video <video_path> --keypoints_csv <output_csv_path> [--model <model_path>] [--log] [--progress]
```

### 2. De-identify Videos

After extracting keypoints, you can de-identify the video in two ways:

#### Face Blur Only

Blur only the faces in the video:

```bash
python video_deid.py --operation_type deid --video <video_path> --keypoints_csv <keypoints_csv> --output <output_path> [--model <model_path>] [--log] [--progress]
```

#### Complete De-identification

Completely blur the entire video and overlay skeleton keypoints:

```bash
python video_deid.py --operation_type deid --video <video_path> --keypoints_csv <keypoints_csv> --output <output_path> --complete_deid [--log] [--progress]
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
