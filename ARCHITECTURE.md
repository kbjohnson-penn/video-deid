# Video De-identification Tool Architecture

## Overview

The Video De-identification Tool is designed to provide a comprehensive solution for de-identifying faces in videos through blurring techniques. It uses computer vision and machine learning to detect facial keypoints and applies various blur effects to protect privacy while maintaining the contextual information of body movements and audio.

## System Architecture

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
├── build/                  # Build artifacts
├── runs/                   # Output directory for runs
├── setup.py                # Package installation
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

## Main Components

1. **CLI Module** (`cli.py`):

   - Parses command-line arguments
   - Orchestrates the workflow between extraction and de-identification
   - Handles logging and progress reporting
   - Manages file paths and temporary files

2. **Blur Package** (`blur/`): Contains functionality for blurring faces in videos:

   - `core.py`: Coordinates the blur process and manages the pipeline
   - `techniques.py`: Implements different blur techniques (Gaussian, median, pixelation)
   - `tracking.py`: Implements Kalman filter tracking for stable face detection across frames
   - `processing.py`: Processes frames in batches for optimal performance

3. **Keypoints Package** (`keypoints/`): Handles pose detection and keypoint extraction:

   - `extraction.py`: Extracts body and facial keypoints using YOLO pose models

4. **De-identification Package** (`deid/`): Contains complete de-identification functionality:

   - `complete.py`: Implements full video de-identification with blurring and skeleton visualization

5. **Utils Package** (`utils/`): Contains utility functions:

   - `file_utils.py`: File and directory operations, temporary file management
   - `keypoint_utils.py`: Keypoint interpolation, filtering, and manipulation
   - `time_utils.py`: Timing and performance measurement utilities
   - `progress.py`: Progress reporting with tqdm integration

6. **Audio Module** (`audio.py`):

   - Extracts audio from videos
   - Combines processed video with original audio
   - Ensures audio quality is preserved during de-identification

7. **Configuration Module** (`config.py`):
   - Centralizes configuration parameters
   - Defines constants for blur intensities, batch sizes, and tracking parameters
   - Enables customization without code changes

## Data Flow

1. **Keypoint Extraction Phase**:

   - Input video is processed frame by frame
   - YOLO model detects people and extracts keypoints
   - Keypoints are saved to a CSV file with tracking information

2. **Keypoint Processing**:

   - Keypoints are loaded from CSV
   - Missing keypoints are interpolated for continuity
   - Optional Kalman filtering for smoothing motion

3. **De-identification Phase**:
   - Input video is processed in batches
   - For face blur: Only facial areas are blurred based on keypoints
   - For complete de-id: Entire frame is blurred and skeleton is overlaid
   - Processed frames are combined with original audio
   - Final video is saved to output path

## Technical Design Decisions

- **Batch Processing**: Frames are processed in configurable batches for better memory usage and performance
- **Modular Architecture**: Components are encapsulated with clear interfaces for maintainability
- **Kalman Filtering**: Advanced tracking to ensure stable blur regions across frames
- **Centralized Configuration**: All parameters are defined in config.py for easy tuning
- **Progress Reporting**: Real-time feedback for long-running operations
- **Error Handling**: Comprehensive error handling with informative messages

## Performance Considerations

- **Batch Processing**: Optimizes memory usage by processing frames in batches
- **Configurable Parameters**: Blur intensity and kernel size can be tuned for performance
- **Memory Management**: Temporary files used for large videos to prevent memory issues
- **Progress Reporting**: Gives feedback during long-running operations

## Future Improvements

- **GPU Acceleration**: Add CUDA support for faster processing
- **Parallel Processing**: Implement multiprocessing for frame-level parallelism
- **Additional Blur Techniques**: Implement more sophisticated anonymization methods
- **Test Coverage**: Add comprehensive unit and integration tests
- **CI/CD Pipeline**: Set up automated testing and deployment
- **Video Streaming**: Support for real-time processing of video streams
