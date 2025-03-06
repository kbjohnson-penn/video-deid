"""
Audio processing functionality for video de-identification
"""
import logging
import subprocess
import tempfile
import os
from pathlib import Path


def extract_audio(video_path):
    """
    Extract audio from a video file and save it to a temporary WAV file.
    
    Parameters:
    - video_path (str): Path to the video file
    
    Returns:
    - str: Path to the temporary WAV file, or None if extraction fails
    """
    try:
        # Create a temp file for the extracted audio
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Extract audio using ffmpeg
        logging.info(f"Extracting audio from {video_path} to {temp_audio_path}")
        subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", temp_audio_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return temp_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return None
    except Exception as e:
        logging.error(f"Unexpected error during audio extraction: {str(e)}")
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return None


def combine_audio_video(original_video, processed_video, output_path):
    """
    Combine audio from original video with processed video.
    If the original video has no audio, just copy the processed video.
    
    Parameters:
    - original_video (str): Path to the original video with audio
    - processed_video (str): Path to the processed video without audio
    - output_path (str): Path to save the final video
    
    Returns:
    - bool: True if successful, False otherwise
    """
    # Convert paths to Path objects
    original_video = Path(original_video)
    processed_video = Path(processed_video)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract audio from original video
    temp_audio_path = extract_audio(original_video)
    
    try:
        if temp_audio_path:
            # Combine audio with processed video
            logging.info(f"Combining audio with processed video to {output_path}")
            subprocess.run(
                ["ffmpeg", "-i", str(processed_video), "-i", temp_audio_path, 
                 "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", 
                 "-shortest", str(output_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            # No audio extracted, just copy the processed video
            logging.info(f"No audio found in original video, copying processed video to {output_path}")
            subprocess.run(
                ["ffmpeg", "-i", str(processed_video), "-c", "copy", str(output_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error combining audio/video: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during audio/video combination: {str(e)}")
        return False
    finally:
        # Clean up temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
            logging.debug(f"Removed temporary audio file: {temp_audio_path}")