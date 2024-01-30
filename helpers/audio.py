from moviepy.editor import VideoFileClip


def combine_audio_video(original_video_path, processed_video_path, output_path):
    # Create a VideoFileClip object for the original and processed videos
    original_video = VideoFileClip(original_video_path)
    processed_video = VideoFileClip(processed_video_path)

    # Set the audio of the processed video to be the audio of the original video
    processed_video_with_audio = processed_video.set_audio(
        original_video.audio)

    # Write the result to a file
    processed_video_with_audio.write_videofile(output_path, codec='libx264')
