"""
Video-DeID: A package for de-identifying faces in videos

This package provides tools for facial keypoint extraction and
de-identification through blurring.
"""
from .blur import process_video
from .keypoints import extract_keypoints_and_save
from .deid import blur_video, process_blurred_video
from .audio import combine_audio_video
from .databricks import extract_keypoints_in_databricks, copy_from_volume_to_local, write_dataframe_to_volume, write_single_csv_file

__version__ = "1.0.0"

__all__ = [
    'process_video',
    'extract_keypoints_and_save',
    'blur_video',
    'process_blurred_video',
    'combine_audio_video',
    'extract_keypoints_in_databricks'
]
