from ultralytics import YOLO
import numpy as np

# Yolo model
yolo_model = 'yolov8n-pose.pt'

# Video source
video_source = '/home/CSI_03.02.18_Trexler_01_TRIMMED.mp4'

# Load a pretrained YOLOv8n model
model = YOLO(yolo_model)

# Run inference on the source
results = model(video_source, show_labels=True, save_txt=True, save_frames=True, save=True, save_conf=True)  # List of Results objects

all_keypoints = []
# Process results 
for result in results:
    # Extracting keypoints
    keypoints = result.keypoints  # Keypoints object for pose outputs
    if keypoints.xy is not None:
        # Convert the tensor to a numpy array and then to a list
        keypoints_list = keypoints.xy.cpu().numpy().tolist()
        all_keypoints.append(keypoints_list)

# Write the keypoints to a file
with open('keypoints_xy.txt', 'w') as f:
    for keypoints in all_keypoints:
        # Convert list to string and write to file
        keypoints_str = ', '.join([str(kp) for kp in keypoints])
        f.write(keypoints_str + '\n')
