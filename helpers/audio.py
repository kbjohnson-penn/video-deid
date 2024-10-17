import subprocess
import tempfile
import os
import logging


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

    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name

    try:
        # Extract audio from the original video to the temporary audio file
        logging.info('Extracting audio from the original video.')
        command_extract_audio = [
            'ffmpeg', '-y',  # '-y' to overwrite if the file already exists
            '-i', original_video_path,
            '-q:a', '0',
            '-map', '0:a',
            temp_audio_path
        ]
        subprocess.run(command_extract_audio, check=True)

        # Check if the audio file is valid and not empty
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) == 0:
            logging.error("Extracted audio file is empty.")
            raise ValueError(
                "The audio extraction failed; extracted audio file is empty.")

        # Combine the extracted audio with the processed video
        logging.info('Combining extracted audio with the processed video.')
        command_combine = [
            'ffmpeg', '-y',  # '-y' to overwrite the output file
            '-i', processed_video_path,
            '-i', temp_audio_path,
            '-c:v', 'copy',
            '-map', '0:v', '-map', '1:a',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            output_path
        ]
        subprocess.run(command_combine, check=True)

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error occurred: {e}")
        raise
    except ValueError as e:
        logging.error(e)
        raise
    finally:
        # Remove the temporary audio file
        logging.info('Removing temporary audio file.')
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
