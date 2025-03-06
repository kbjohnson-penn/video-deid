"""
Time-related utility functions
"""

def calculate_time(frame_number, frame_rate):
    """
    Calculates the time in the video given the frame number and the frame rate.
    
    Parameters:
    - frame_number (int): The frame number
    - frame_rate (float): The frame rate (frames per second)
    
    Returns:
    - float: The time in seconds
    """
    return frame_number / frame_rate