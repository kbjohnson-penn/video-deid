# Video De-identification Tool Architecture

## Overview

The Video De-identification Tool is designed to provide a comprehensive solution for de-identifying faces in videos through blurring. It uses computer vision techniques to detect facial keypoints and applies blur effects to protect privacy.

## Architecture

The codebase follows a modular design with clear separation of concerns:

```
video-deid/
├── src/                    # Source code directory
│   └── video_deid/         # Main package
│       ├── __init__.py     # Package initialization
│       ├── cli.py          # Command-line interface
│       ├── config.py       # Centralized configuration
│       ├── blur/           # Blur functionality
│       │   ├── __init__.py
│       │   ├── core.py     # Main blur process
│       │   ├── techniques.py # Blur implementations
│       │   ├── tracking.py # Kalman tracking
│       │   └── processing.py # Frame processing
│       ├── keypoints/      # Keypoint extraction 
│       │   ├── __init__.py
│       │   └── extraction.py # Keypoint extraction with YOLO
│       ├── deid/           # De-identification functionality
│       │   ├── __init__.py
│       │   └── complete.py # Complete video de-identification
│       ├── utils/          # Utility functions
│       │   ├── __init__.py
│       │   ├── file_utils.py   # File operations
│       │   ├── keypoint_utils.py # Keypoint handling
│       │   ├── time_utils.py  # Time-related utilities
│       │   └── progress.py    # Progress reporting
│       └── audio.py        # Audio processing
├── scripts/                # Shell scripts for batch processing
├── data/                   # Sample data
├── setup.py                # Package installation
└── README.md               # Project documentation
```

## Main Components

1. **CLI Module** (`cli.py`): Handles command-line arguments and coordinates the de-identification workflow.

2. **Blur Package** (`blur/`): Contains functionality for blurring faces in videos:
   - `core.py`: Coordinates the blur process
   - `techniques.py`: Implements different blur techniques
   - `tracking.py`: Implements Kalman filter tracking for faces
   - `processing.py`: Handles frame-by-frame processing

3. **Keypoints Package** (`keypoints/`): Handles keypoint extraction:
   - `extraction.py`: Extracts keypoints using YOLO models

4. **De-identification Package** (`deid/`): Contains de-identification functionality:
   - `complete.py`: Implements complete video de-identification with blurring and keypoint visualization

5. **Utils Package** (`utils/`): Contains utility functions:
   - `file_utils.py`: File and directory operations
   - `keypoint_utils.py`: Keypoint processing and manipulation
   - `time_utils.py`: Time-related utilities
   - `progress.py`: Progress reporting

6. **Audio Module** (`audio.py`): Handles audio extraction and combination.

7. **Configuration Module** (`config.py`): Centralizes configuration parameters.

## Data Flow

1. User input is processed via the CLI module.
2. The CLI module coordinates the workflow based on the operation type:
   - For keypoint extraction, the `extract_keypoints_and_save` function is called.
   - For de-identification, the process involves loading keypoints, interpolation, and video processing.
3. The processed video is saved to the output path.

## Design Decisions

- **Modularity**: The codebase is organized into modules with clear responsibilities, making it easier to maintain and extend.
- **Clean Package Structure**: Functionality is organized into logical subpackages (blur, keypoints, deid, utils).
- **No Backward Compatibility**: Backward compatibility has been removed to eliminate redundant code.
- **Centralized Configuration**: Configuration parameters are centralized in `config.py`.
- **Proper Error Handling**: Comprehensive error handling is implemented throughout the codebase.
- **Progress Reporting**: Progress reporting is available for long-running operations.
- **Flexible Blur Techniques**: The blur techniques are modular and can be extended with new methods.

## Future Improvements

- Add unit tests for core functionality
- Implement more blur techniques
- Add support for parallel processing of video frames
- Create a web interface for non-technical users
- Improve documentation and examples
- Set up CI/CD pipelines for automated testing and deployment