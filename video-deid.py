import cv2
import numpy as np
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_keypoints(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                yield ast.literal_eval(line.strip())
    except FileNotFoundError:
        logging.error(f"Keypoints file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading keypoints file: {e}")
        raise


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
                radius = max(x_max - x_min, y_max - y_min)
                if radius <= 50:
                    radius = 90
            else:
                # Use a default radius for the blur area when only one keypoint is available
                default_radius = 100  # Adjust as needed
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


def process_video(video_path, keypoints_file_path, output_path):
    logging.info(f"Starting video processing for {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Unable to open video file: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        keypoints_generator = read_keypoints(keypoints_file_path)
        frame_count = 0
        update_interval = 100  # Adjust as needed

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("No more frames to read from video.")
                break

            frame_count += 1
            if frame_count % update_interval == 0:
                percentage = (frame_count / total_frames) * 100
                logging.info(
                    f"Processing frame {frame_count}/{total_frames} ({percentage:.2f}%)")

            try:
                keypoints = next(keypoints_generator)
            except StopIteration:
                logging.info("No more keypoints available.")
                break
            except Exception as e:
                logging.error(
                    f"Error processing keypoints for frame {frame_count}: {e}")
                break

            try:
                processed_frame = blur_face_circle(frame, keypoints)
                out.write(processed_frame)
            except Exception as e:
                logging.error(
                    f"Error applying blur to frame {frame_count}: {e}")
                break

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        if cap:
            cap.release()
        if out:
            out.release()
        logging.info("Released video resources.")

    logging.info(f"Video processing completed. Output saved to {output_path}")


# Usage
video_path = '/Users/mopidevi/Downloads/CSI_3.10.18_Sweeney_03.mp4'
keypoints_file_path = '/Users/mopidevi/Workspace/projects/video-deid/keypoints_xy.txt'
output_path = '/Users/mopidevi/Workspace/projects/video-deid/output_1.mp4'

process_video(video_path, keypoints_file_path, output_path)
