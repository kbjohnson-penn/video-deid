import subprocess


def combine_audio_video(original_video_path, processed_video_path, output_path):
    """
    Combines the audio of the original video with the processed video using ffmpeg.

    Parameters:
    - original_video_path (str): The path to the original video.
    - processed_video_path (str): The path to the processed video.
    - output_path (str): The path to save the output video.

    Returns:
    - None
    """

    # Extract audio from the original video to a temporary audio file
    temp_audio_path = "temp_audio.mp3"
    command_extract_audio = [
        'ffmpeg', '-i', original_video_path, '-q:a', '0', '-map', 'a', temp_audio_path
    ]
    subprocess.run(command_extract_audio, check=True)

    # Combine the extracted audio with the processed video
    command_combine = [
        'ffmpeg', '-i', processed_video_path, '-i', temp_audio_path, '-c:v', 'copy',
        '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac', '-strict', 'experimental',
        '-shortest', output_path
    ]
    subprocess.run(command_combine, check=True)

    # Optionally remove the temporary audio file
    subprocess.run(['rm', temp_audio_path])
