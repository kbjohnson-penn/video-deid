import cv2
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(filename='video_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def draw_face_bbox_and_keypoints(frame, keypoints):
    keypoints = np.array(keypoints)  # Convert keypoints to a NumPy array

    if keypoints.shape[0] > 0:
        # Convert normalized coordinates to pixel coordinates
        keypoints[:, 0] *= frame.shape[1]
        keypoints[:, 1] *= frame.shape[0]

        # Filter out missing keypoints
        valid_keypoints = keypoints[(keypoints[:, 0] != 0) & (keypoints[:, 1] != 0)]

        if valid_keypoints.shape[0] > 0:
            margin = 30  # You can adjust this value as needed

            if valid_keypoints.shape[0] > 1:
                x_min, y_min = np.min(valid_keypoints[:, :2], axis=0).astype(int) - margin
                x_max, y_max = np.max(valid_keypoints[:, :2], axis=0).astype(int) + margin
            else:
                # If there's only one keypoint, create a bounding box of fixed size around it
                x, y = valid_keypoints[0, :2].astype(int)
                box_size = 40  # You can adjust this value as needed
                x_min = max(0, x - box_size)
                y_min = max(0, y - box_size)
                x_max = min(frame.shape[1], x + box_size)
                y_max = min(frame.shape[0], y + box_size)

            # Ensure the bounding box coordinates are within the frame boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            for x, y, _ in valid_keypoints.astype(int):
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)
        else:
            logging.warning('No valid keypoints detected in this frame.')
    else:
        logging.warning('No keypoints detected in this frame.')

    return frame

video_path = '/Users/mopidevi/Workspace/projects/video-deid/CSI_03.02.18_Trexler_01_TRIMMED.mp4'
keypoints_dir = '/Users/mopidevi/Workspace/projects/video-deid/labels'
output_path = '/Users/mopidevi/Workspace/projects/video-deid/output.mp4'

video_name = os.path.splitext(os.path.basename(video_path))[0]

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    logging.error("Cannot open video")
    raise IOError("Cannot open video")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_number = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints_file_name = f"{video_name}_{frame_number}.txt"
    keypoints_file_path = os.path.join(keypoints_dir, keypoints_file_name)

    if os.path.exists(keypoints_file_path):
        with open(keypoints_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = [float(x) for x in line.split()]
                face_keypoints = [(data[i], data[i+1], data[i+2]) for i in range(5, 5+5*3, 3)]
                logging.info(f"Frame {frame_number} keypoints: {face_keypoints}")
                frame = draw_face_bbox_and_keypoints(frame, face_keypoints)
    else:
        logging.warning(f"Keypoints file not found for frame {frame_number}: {keypoints_file_name}")

    if frame is not None:
        out.write(frame)
    else:
        print("Invalid frame")
    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()
