import argparse
import os
from ultralytics import YOLO
import csv
import json
import logging


def create_csv(filename):
    """
    Create a csv file with the given filename and write the header to it.

    Parameters:
    - filename: The name of the csv file to be created.

    Returns:
    - None
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        csv_header = ['frame_number', 'person_id',
                      'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        for i in range(0, 17):
            csv_header.append('x_' + str(i))
            csv_header.append('y_' + str(i))
            csv_header.append('c_' + str(i))
        writer.writerow(csv_header)


def write_to_csv(data, filename):
    """
    Write the given data to the csv file with the given filename.

    Parameters:
    - data: The data to be written to the csv file.
    - filename: The name of the csv file to write the data to.

    Returns:
    - None
    """
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def prepare_and_write_data(frame_number, person_id, bbox, keypoints, csv_filename):
    csv_data = [frame_number, person_id, bbox['x1'],
                bbox['y1'], bbox['x2'], bbox['y2']]
    x = keypoints['x']
    y = keypoints['y']
    c = keypoints['visible']
    for i in range(0, 17):
        csv_data.extend([x[i], y[i], c[i]])
    write_to_csv(csv_data, csv_filename)
    logging.info(
        f"Data for frame {frame_number}, person {person_id} written to CSV.")


def write_no_detections_data(frame_number, csv_filename):
    """
    Write a row of zeros to the CSV file when there are no detections in a frame.

    Parameters:
    - frame_number: The frame number where no detections were found.
    - csv_filename: The name of the CSV file to write the data to.

    Returns:
    - None
    """
    no_detections_data = [frame_number, 'No detections', 0, 0, 0, 0]
    for _ in range(0, 17):
        no_detections_data.extend([0, 0, 0])
    write_to_csv(no_detections_data, csv_filename)
    logging.info(
        f"No detections data for frame {frame_number} written to CSV.")


def extract_keypoints_and_save(yolov8_model, video_source, csv_filename):
    """
    Extract the keypoints and bounding box coordinates from the video source using the YOLOv8 model, track the person across frames,
    and save them to a csv file. Interpolate positions for missed detections.

    Parameters:
    - yolov8_model: The YOLOv8 model to use for extracting the keypoints and bounding box coordinates.
    - video_source: The video source to extract the keypoints and bounding box coordinates from.
    - csv_filename: The name of the csv file to save the keypoints and bounding box coordinates to.

    Returns:
    - None
    """

    # Improved logging and error handling
    try:
        # Create the csv file and write the header to it
        create_csv(csv_filename)
        logging.info(f"CSV file {csv_filename} created and header written.")

        # Load the YOLOv8 model
        logging.info("Loading YOLOv8 model...")
        model = YOLO(yolov8_model)
        logging.info("YOLOv8 model loaded.")

        # Track the keypoints and bounding box coordinates in the video source
        logging.info(f"Starting tracking on video source: {video_source}")
        results = model.track(source=video_source, stream=True)

        if results is None:
            logging.error("No results from the YOLOv8 model.")
            return

        # Iterate over each frame in the video source
        frame_number = 0
        for result in results:
            logging.info(f"Processing frame {frame_number}")
            # Convert the result to json format
            try:
                json_data = json.loads(result.tojson())
            except Exception as e:
                logging.error(
                    f"Error parsing JSON data for frame {frame_number}: {e}")
                frame_number += 1
                continue

            # Check if there are any detections in the frame
            if not json_data:
                logging.info(
                    f"No detections in frame {frame_number}. Logging frame and person ID placeholder.")
                # Log the frame number and a placeholder for person ID
                logging.info(
                    f"Frame: {frame_number}, Person ID: 'No detections'")
                # Write no detections data to CSV
                write_no_detections_data(frame_number, csv_filename)
                frame_number += 1
                continue

            # Iterate over each person in the frame
            for person in json_data:
                if 'track_id' not in person:
                    logging.warning(
                        f"'track_id' not found for a person in frame {frame_number}. Skipping this person.")
                    continue

                person_id = person['track_id']
                bbox = person['box']
                keypoints = person['keypoints']

                # Prepare and write data to CSV
                prepare_and_write_data(
                    frame_number, person_id, bbox, keypoints, csv_filename)

            frame_number += 1
            logging.info(f"Completed processing for frame {frame_number}.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Extract keypoints and save to CSV.')
    parser.add_argument('video_source', type=str,
                        help='Path to the video source')
    parser.add_argument('csv_filename', type=str,
                        help='Path to the CSV filename')
    args = parser.parse_args()

    video_source = args.video_source
    csv_filename = args.csv_filename

    log_filename = os.path.splitext(os.path.basename(csv_filename))[0] + '.log'
    log_filepath = os.path.join(os.path.dirname(csv_filename), log_filename)

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filepath, mode='w'),
                        ])

    yolov8_model = "yolov8x-pose-p6.pt"

    # Extract keypoints and save to CSV
    extract_keypoints_and_save(yolov8_model, video_source, csv_filename)
    logging.info("Completed keypoint extraction.")
