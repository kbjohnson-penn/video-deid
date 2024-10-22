import cv2
import logging
from helpers.utils import create_progress_bar


def blur_entire_frame(frame):
    kernel_size = (151, 151)
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)
    return blurred_frame


def blur_video(video_source, temp_blurred_video_path, show_progress=False):
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logging.error(f"Failed to open video source {video_source}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_blurred_video_path,
                              fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = create_progress_bar(
            total_frames, "Blurring video", show_progress)

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            blurred_frame = blur_entire_frame(frame)
            out.write(blurred_frame)
            if progress_bar:
                progress_bar.update(1)

    except Exception as e:
        logging.error(f"Failed to process video {video_source}: {e}")
    finally:
        # Ensure resources are always released
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        if progress_bar:
            progress_bar.close()
        cv2.destroyAllWindows()
        logging.info(f"Blurred video saved to {temp_blurred_video_path}")


def draw_annotations(frame, keypoints):
    skeleton = [
        ((0, 1), 'green'), ((0, 2), 'green'), ((1, 3), 'green'), ((2, 4), 'green'),
        ((3, 5), 'blue'), ((4, 6), 'blue'), ((5, 6), 'blue'),
        ((5, 7), 'blue'), ((7, 9), 'blue'), ((6, 8), 'blue'), ((8, 10), 'blue'),
        ((5, 11), 'orange'), ((6, 12), 'orange'), ((11, 12), 'orange'),
        ((11, 13), 'orange'), ((13, 15), 'orange'), ((
            12, 14), 'orange'), ((14, 16), 'orange')
    ]

    for i, (x, y) in enumerate(keypoints):
        if not (x == 0.0 and y == 0.0):
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        else:
            logging.debug(f"Skipping keypoint {i} because it's (0.0, 0.0)")

    color_map = {'green': (0, 255, 0), 'blue': (
        0, 0, 255), 'orange': (0, 165, 255)}
    for (start_point, end_point), color in skeleton:
        start_x, start_y = keypoints[start_point]
        end_x, end_y = keypoints[end_point]
        if not (start_x == 0.0 and start_y == 0.0) and not (end_x == 0.0 and end_y == 0.0):
            cv2.line(frame, (int(start_x), int(start_y)),
                     (int(end_x), int(end_y)), color_map[color], 2)
        else:
            logging.debug(
                f"Skipping line from {start_point} to {end_point} due to invalid keypoints.")

    return frame


def process_blurred_video(blurred_video_source, df, output_video_path, show_progress=False):
    try:
        cap = cv2.VideoCapture(blurred_video_source)
        if not cap.isOpened():
            logging.error(
                f"Failed to open video source {blurred_video_source}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = create_progress_bar(
            total_frames, "Applying keypoints on blurred video", show_progress)

        for frame_number in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_copy = frame.copy()
            for index, row in df[(df['frame_number'] == frame_number)].iterrows():
                keypoints = [(row[f'x_{i}'], row[f'y_{i}'])
                             for i in range(0, 17)]
                frame_copy = draw_annotations(frame_copy, keypoints)

            out.write(frame_copy)
            if progress_bar:
                progress_bar.update(1)
            logging.info(f"Processed frame {frame_number}")

    except Exception as e:
        logging.error(f"Failed to process video {blurred_video_source}: {e}")
    finally:
        # Ensure resources are always released
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        if progress_bar:
            progress_bar.close()
        cv2.destroyAllWindows()
        logging.info("Completed processing all frames.")
