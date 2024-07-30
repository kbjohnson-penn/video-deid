import cv2
import pandas as pd
import logging
import sys
import tempfile
import os

def blur_entire_frame(frame):
    kernel_size = (151, 151)
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)
    return blurred_frame

def blur_video(video_source, temp_blurred_video_path):
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logging.error(f"Failed to open video source {video_source}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_blurred_video_path, fourcc, fps, (width, height))
    except Exception as e:
        logging.error(f"Failed to process video {video_source}: {e}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blurred_frame = blur_entire_frame(frame)
        out.write(blurred_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"Blurred video saved to {temp_blurred_video_path}")

def draw_annotations(frame, keypoints):
    skeleton = [
        ((0, 1), 'green'), ((0, 2), 'green'), ((1, 3), 'green'), ((2, 4), 'green'), 
        ((3, 5), 'blue'), ((4, 6), 'blue'), ((5, 6), 'blue'), 
        ((5, 7), 'blue'), ((7, 9), 'blue'), ((6, 8), 'blue'), ((8, 10), 'blue'), 
        ((5, 11), 'orange'), ((6, 12), 'orange'), ((11, 12), 'orange'), 
        ((11, 13), 'orange'), ((13, 15), 'orange'), ((12, 14), 'orange'), ((14, 16), 'orange')
    ]

    for i, (x, y) in enumerate(keypoints):
        if not (x == 0.0 and y == 0.0):
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        else:
            logging.debug(f"Skipping keypoint {i} because it's (0.0, 0.0)")

    color_map = {'green': (0, 255, 0), 'blue': (0, 0, 255), 'orange': (0, 165, 255)}
    for (start_point, end_point), color in skeleton:
        start_x, start_y = keypoints[start_point]
        end_x, end_y = keypoints[end_point]
        if not (start_x == 0.0 and start_y == 0.0) and not (end_x == 0.0 and end_y == 0.0):
            cv2.line(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color_map[color], 2)
        else:
            logging.debug(f"Skipping line from {start_point} to {end_point} due to invalid keypoints.")

    return frame

def process_blurred_video(blurred_video_source, csv_file, output_video_path):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"Failed to read CSV file {csv_file}: {e}")
        return

    try:
        cap = cv2.VideoCapture(blurred_video_source)
        if not cap.isOpened():
            logging.error(f"Failed to open video source {blurred_video_source}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    except Exception as e:
        logging.error(f"Failed to process video {blurred_video_source}: {e}")
        return

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_copy = frame.copy()
        for index, row in df[(df['frame_number'] == frame_number)].iterrows():
            keypoints = [(row[f'x_{i}'], row[f'y_{i}']) for i in range(0, 17)]
            frame_copy = draw_annotations(frame_copy, keypoints)

        out.write(frame_copy)
        frame_number += 1
        logging.info(f"Processed frame {frame_number}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("Completed processing all frames.")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python deid.py <video_file_path> <video_file_name> <csv_file_path> <output_video_path> <log_file_path>")
        sys.exit(1)

    video_file_path = sys.argv[1]
    video_file_name = sys.argv[2]
    csv_file_path = sys.argv[3]
    output_video_path = sys.argv[4]
    log_file_path = sys.argv[5]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file_path, mode='w')])

    # Blur the entire video and save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_blurred_video:
        temp_blurred_video_path = temp_blurred_video.name

    logging.info(f"Starting video blur -- {video_file_name}.")
    blur_video(video_file_path, temp_blurred_video_path)
    logging.info("Completed video blurring.")

    # Process the blurred video and overlay keypoints
    logging.info("Processing Blurred Video.")
    process_blurred_video(temp_blurred_video_path, csv_file_path, output_video_path)
    logging.info(f"Completed processing all frames -- {video_file_name}.")

    # Remove the temporary file
    os.remove(temp_blurred_video_path)
