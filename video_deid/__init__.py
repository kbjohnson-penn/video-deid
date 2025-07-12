"""
Video-DeID: A package for de-identifying faces in videos

This package provides tools for facial keypoint extraction and
de-identification through blurring.
"""

__version__ = "1.0.0"

# Lazy imports to avoid dependency issues
def _check_dependencies():
    """Check if required dependencies are available."""
    try:
        import cv2
        import numpy as np
        return True
    except ImportError:
        return False

def get_functions():
    """Get available functions (imports modules dynamically)."""
    if not _check_dependencies():
        raise ImportError(
            "Missing required dependencies. Install with:\n"
            "pip install opencv-python numpy\n"
            "or\n"
            "conda install opencv-python numpy"
        )
    
    from .blur import process_video
    from .keypoints import extract_keypoints_and_save
    from .deid import blur_video, process_blurred_video
    from .audio import combine_audio_video
    
    return {
        'process_video': process_video,
        'extract_keypoints_and_save': extract_keypoints_and_save,
        'blur_video': blur_video,
        'process_blurred_video': process_blurred_video,
        'combine_audio_video': combine_audio_video
    }

# Make functions available at module level when dependencies are present
if _check_dependencies():
    try:
        functions = get_functions()
        globals().update(functions)
        __all__ = list(functions.keys())
    except ImportError:
        __all__ = []
else:
    __all__ = []
