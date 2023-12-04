from moviepy.editor import VideoFileClip
from mutagen.mp3 import MP3
import os
import math
import cv2

original_video_file = "/Users/mopidevi/Workspace/projects/video-deid/original_video_CSI_3.10.18_Sweeney_03.mp4"
original_audio_file = "/Users/mopidevi/Workspace/projects/video-deid/original_audio.mp3"
processed_video_file = "/Users/mopidevi/Workspace/projects/video-deid/de-id.mp4"
processed_audio_file = "/Users/mopidevi/Workspace/projects/video-deid/processed_Sweeny_03_audio.mp3"


def check_fps(file_name):
    cam = cv2.VideoCapture(file_name)
    fps = cam.get(cv2.CAP_PROP_FPS)
    return fps


def convert_size(size_bytes):
    """ Convert bytes to a more readable format """
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def video_parameters(video_file):
    # For Video File
    video = VideoFileClip(video_file)
    video_duration = video.duration  # Duration in seconds
    # Size in human-readable format
    video_size = convert_size(os.path.getsize(video_file))
    # Get FPS for Video File
    video_fps = check_fps(video_file)
    return video_duration, video_size, video_fps


def audio_parametes(audio_file):
    # For Audio File
    audio = MP3(audio_file)
    audio_duration = audio.info.length  # Duration in seconds
    # Size in human-readable format
    audio_size = convert_size(os.path.getsize(audio_file))
    return audio_duration, audio_size


original_audio_duration, original_audio_size = audio_parametes(
    original_audio_file)
original_video_duration, original_video_size, original_video_fps = video_parameters(
    original_video_file)

processed_audio_duration, processed_audio_size = audio_parametes(
    processed_audio_file)
processed_video_duration, processed_video_size, processed_video_fps = video_parameters(
    processed_video_file)

print(
    f"Original Video Duration: {original_video_duration} seconds, Size: {original_video_size}, FPS: {original_video_fps}")
print(
    f"Original Audio Duration: {original_audio_duration} seconds, Size: {original_audio_size}")

print(
    f"Processed Video Duration: {processed_video_duration} seconds, Size: {processed_video_size}, FPS: {processed_video_fps}")
print(
    f"Processed Audio Duration: {processed_audio_duration} seconds, Size: {processed_audio_size}")
