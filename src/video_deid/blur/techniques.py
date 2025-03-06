"""
Blur techniques for facial de-identification
"""
import cv2
import numpy as np


def apply_circular_blur(frame, x_min, y_min, x_max, y_max):
    """
    Applies a circular blur to a region in a frame.

    Parameters:
    - frame (np.array): The frame to apply the blur to.
    - x_min, y_min, x_max, y_max (int): The coordinates of the region to blur.

    Returns:
    - np.array: The frame with the blurred region.
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Calculate center and radius
    center_x = int((x_min + x_max) // 2)
    center_y = int((y_min + y_max) // 2)
    radius = int(max((x_max - x_min) // 2, (y_max - y_min) // 2)) * 2
    
    # Define ROI with padding
    padding = radius // 2
    roi_x_min = max(0, center_x - radius - padding)
    roi_y_min = max(0, center_y - radius - padding)
    roi_x_max = min(width, center_x + radius + padding)
    roi_y_max = min(height, center_y + radius + padding)
    
    # Extract ROI
    roi = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
    
    # Create mask only for ROI
    roi_mask = np.zeros(roi.shape[:2], dtype="uint8")
    cv2.circle(
        roi_mask, 
        (center_x - roi_x_min, center_y - roi_y_min), 
        radius, 
        255, 
        -1
    )
    
    # Use smaller kernel size for efficiency
    kernel_size = min(75, max(21, radius // 2))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd
    
    # Blur only the ROI
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 15)
    
    # Apply the blur only to masked area
    np.copyto(roi, blurred_roi, where=roi_mask[:, :, None] == 255)
    
    # Update the frame with the modified ROI
    frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = roi
    
    return frame


def apply_full_frame_blur(frame, kernel_size=25):
    """
    Applies a light Gaussian blur to the entire frame for privacy protection.
    
    Parameters:
    - frame (np.array): The frame to blur
    - kernel_size (int): Size of the Gaussian kernel
    
    Returns:
    - np.array: The blurred frame
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 10)


def draw_bounding_box_and_keypoints(frame, keypoints, x_min, y_min, x_max, y_max):
    """
    Draws a bounding box and keypoints on a frame.

    Parameters:
    - frame (np.array): The frame to draw on.
    - keypoints (list): The keypoints to draw.
    - x_min, y_min, x_max, y_max (int): The coordinates of the bounding box.

    Returns:
    - np.array: The frame with the bounding box and keypoints drawn on it.
    """
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    for keypoint in keypoints:
        cv2.circle(frame, (int(keypoint[0]), int(
            keypoint[1])), 2, (0, 0, 255), -1)

    return frame