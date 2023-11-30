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


def blur_face_rectangle(frame, keypoints_list):
    if len(keypoints_list) == 17:
        persons_keypoints = [keypoints_list]
    else:
        persons_keypoints = keypoints_list

    for keypoints in persons_keypoints:
        face_keypoints = [kp for kp in keypoints[:5]
                          if isinstance(kp, list) and len(kp) == 2]
        filtered_keypoints = [kp for kp in face_keypoints if not (
            kp[0] == 0.0 and kp[1] == 0.0)]

        if filtered_keypoints:
            visible_face_keypoints = np.array(filtered_keypoints)

            if len(filtered_keypoints) >= 2:
                # Calculate bounding box for multiple keypoints
                x_min = int(min(visible_face_keypoints[:, 0]))
                y_min = int(min(visible_face_keypoints[:, 1]))
                x_max = int(max(visible_face_keypoints[:, 0]))
                y_max = int(max(visible_face_keypoints[:, 1]))
            else:
                # Use a default size for the blur area when only one keypoint is available
                default_size = 100  # This can be adjusted
                kp = visible_face_keypoints[0]
                x_min = max(0, int(kp[0] - default_size / 2))
                y_min = max(0, int(kp[1] - default_size / 2))
                x_max = min(frame.shape[1], int(kp[0] + default_size / 2))
                y_max = min(frame.shape[0], int(kp[1] + default_size / 2))

            face_region = frame[y_min:y_max, x_min:x_max]
            if face_region.size > 0:
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y_min:y_max, x_min:x_max] = blurred_face

    return frame


def blur_face_circle(frame, keypoints_list):
    if len(keypoints_list) == 17:
        persons_keypoints = [keypoints_list]
    else:
        persons_keypoints = keypoints_list

    for keypoints in persons_keypoints:
        face_keypoints = [kp for kp in keypoints[:5]
                          if isinstance(kp, list) and len(kp) == 2]
        filtered_keypoints = [kp for kp in face_keypoints if not (
            kp[0] == 0.0 and kp[1] == 0.0)]

        if filtered_keypoints:
            visible_face_keypoints = np.array(filtered_keypoints)
            center = None
            radius = None

            if len(filtered_keypoints) >= 2:
                x_min = int(min(visible_face_keypoints[:, 0]))
                y_min = int(min(visible_face_keypoints[:, 1]))
                x_max = int(max(visible_face_keypoints[:, 0]))
                y_max = int(max(visible_face_keypoints[:, 1]))
                center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                radius = max(x_max - x_min, y_max - y_min) // 2
            else:
                # Use a default radius for the blur area when only one keypoint is available
                default_radius = 50  # Adjust as needed
                kp = visible_face_keypoints[0]
                center = (int(kp[0]), int(kp[1]))
                radius = default_radius

            # Create a mask for the circular region
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, center, radius, 255, -1)

            # Blur the entire frame and mask the circular region
            blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)
            frame = np.where(mask[:, :, None] == 255, blurred_frame, frame)

    return frame


def join_frames(frames, width, height, frame_rate, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc,
                          frame_rate, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


# Path to Keypoints and Video
keypoints_file_path = '/Users/mopidevi/Workspace/projects/video-deid/keypoints_xy.txt'
video_path = '/Users/mopidevi/Workspace/projects/video-deid/CSI_03.02.18_Trexler_01_TRIMMED.mp4'
# output_path = '/Users/mopidevi/Workspace/projects/video-deid/deid.mp4'
output_path = '/Users/mopidevi/Workspace/projects/video-deid/deid-cir.mp4'
# output_path = '/Users/mopidevi/Workspace/projects/video-deid/debug.mp4'

video_keypoints = read_keypoints(keypoints_file_path)
video_frames, width, height, frame_rate = load_video(video_path)


if len(video_keypoints) == len(video_frames):
    processed_frames = []
    for idx in range(len(video_keypoints)):
        processed_frame = blur_face_circle(
            video_frames[idx], video_keypoints[idx])
        processed_frames.append(processed_frame)
    join_frames(processed_frames, width, height, frame_rate, output_path)
