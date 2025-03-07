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
    Extract audio from a video file and save it to a temporary audio file.
    
    Parameters:
    - video_path (str): Path to the video file
    
    Returns:
    - str: Path to the temporary audio file, or None if extraction fails
    """
    try:
        # Create a temp file for the extracted audio (using mp3 instead of wav for better compatibility)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Check if the video actually has an audio stream
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", 
                    "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)]
        
        probe_result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True
        )
        
        # If there's no audio stream, return None
        if not probe_result.stdout.strip():
            logging.info(f"No audio stream found in {video_path}")
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return None
            
        # Extract audio using ffmpeg
        logging.info(f"Extracting audio from {video_path} to {temp_audio_path}")
        
        # Use a more compatible and safer extraction method
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "libmp3lame", 
             "-q:a", "4", temp_audio_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Verify the extracted audio file exists and has content
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            return temp_audio_path
        else:
            logging.error(f"Failed to extract audio: Output file is empty or missing")
            return None
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
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
    
    try:
        # Check if processed video exists
        if not processed_video.exists():
            logging.error(f"Processed video file not found: {processed_video}")
            return False
        
        # Extract audio from original video
        temp_audio_path = extract_audio(original_video)
        
        if temp_audio_path:
            # Combine audio with processed video
            logging.info(f"Combining audio with processed video to {output_path}")
            
            # Use the -y flag to overwrite output if it exists
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(processed_video), "-i", temp_audio_path,
                 "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                 "-shortest", str(output_path)],
                check=False,  # Don't raise exception, we'll handle errors manually
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if result.returncode != 0:
                logging.error(f"Error combining audio/video: {result.stderr.decode()}")
                # Fall back to copying the processed video without audio
                logging.info(f"Falling back to copying processed video without audio to {output_path}")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(processed_video), "-c", "copy", str(output_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        else:
            # No audio extracted, just copy the processed video
            logging.info(f"No audio found in original video, copying processed video to {output_path}")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(processed_video), "-c", "copy", str(output_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Verify the output file exists and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            logging.error(f"Output file is empty or missing: {output_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in video processing: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during audio/video processing: {str(e)}")
        return False
    finally:
        # Clean up temporary audio file
        if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
            logging.debug(f"Removed temporary audio file: {temp_audio_path}")