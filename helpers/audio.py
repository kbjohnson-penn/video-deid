from moviepy.editor import VideoFileClip


def combine_audio_video(original_video_path, processed_video_path, output_path):
    """
    Combines the audio of the original video with the processed video.

    Parameters:
    - original_video_path (str): The path to the original video.
    - processed_video_path (str): The path to the processed video.
    - output_path (str): The path to save the output video.

    Returns:
    - None
    """

    # Create a VideoFileClip object for the original and processed videos
    original_video = VideoFileClip(original_video_path)
    processed_video = VideoFileClip(processed_video_path)

    # Set the audio of the processed video to be the audio of the original video
    processed_video_with_audio = processed_video.set_audio(
        original_video.audio)

    # Write the result to a file
    processed_video_with_audio.write_videofile(output_path, codec='libx264')
