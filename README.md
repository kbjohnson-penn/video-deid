# Video De-Identification

This project is a Python script designed to process videos by blurring detected faces, thereby anonymizing individuals in the footage. It leverages facial keypoints to locate faces within each video frame and subsequently applies a blur effect. The application offers features such as saving keypoints as CSV, logging of processing details, saving frames where keypoints were not detected for further analysis, and a progress bar for visual tracking of the processing progress.

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

## Usage

This script processes a video and applies a blur to faces detected in the video. Here's how you can use it:

```bash
python video-deid.py --video <video_path> --output <output_path> [--log] [--show_progress]
```

- `--video` argument is the path to the input video.
- `--output` argument is the path to the output video.
- `--log` argument enables logging.
- `--show_progress` argument shows a progress bar.
