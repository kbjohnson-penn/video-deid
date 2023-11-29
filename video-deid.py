import cv2
import numpy as np
import ast  # To convert string representation of list to actual list


def read_keypoints(file_path):
    keypoints = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert string representation of list to an actual list
            keypoints.append(ast.literal_eval(line.strip()))
    return keypoints


def load_video(file_path):
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, width, height, frame_rate


def blur_face(frame, keypoints):
    if len(keypoints) == 17:
        # Modify this based on actual face keypoints
        face_keypoints = np.array(keypoints[:5])
        filtered_keypoints = [kp for kp in face_keypoints if not (
            kp[0] == 0.0 and kp[1] == 0.0)]
        visible_face_keypoints = np.array(filtered_keypoints)

        # Compute the bounding box for the face
        x_min = int(min(visible_face_keypoints[:, 0]))
        y_min = int(min(visible_face_keypoints[:, 1]))
        x_max = int(max(visible_face_keypoints[:, 0]))
        y_max = int(max(visible_face_keypoints[:, 1]))

        # Ensure bounding box is within frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)

        # Blur the face region
        face_region = frame[y_min:y_max, x_min:x_max]
        if face_region.size > 0:    
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y_min:y_max, x_min:x_max] = blurred_face
            # Uncomment for debug
            # frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return frame


def join_frames(frames, width, height, frame_rate, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc,
                          frame_rate, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


# Path to Keypoints and Video
keypoints_file_path = '/Users/mopidevi/Workspace/projects/yolo8/predict/keypoints_xy.txt'
video_path = '/Users/mopidevi/Workspace/projects/yolo8/CSI_03.02.18_Trexler_01_TRIMMED.mp4'
output_path = '/Users/mopidevi/Workspace/projects/yolo8/deid.mp4'
# output_path = '/Users/mopidevi/Workspace/projects/yolo8/debug.mp4'

video_keypoints = read_keypoints(keypoints_file_path)
video_frames, width, height, frame_rate = load_video(video_path)


if len(video_keypoints) == len(video_frames):
    processed_frames = []
    for idx in range(len(video_keypoints)):
        processed_frame = blur_face(video_frames[idx], video_keypoints[idx])
        processed_frames.append(processed_frame)
    join_frames(processed_frames, width, height, frame_rate, output_path)
