"""
Keypoint extraction functionality using YOLO models
"""
from ultralytics import YOLO
import csv
import json
import logging
import threading
import queue
from pathlib import Path
from ..config import KEYPOINTS_DTYPE_SPEC


def create_csv(filename):
    """
    Create a csv file with the given filename and write the header to it.

    Parameters:
    - filename: The name of the csv file to be created.

    Returns:
    - None
    """
    logging.info(f"Creating CSV file {filename} and writing header.")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        csv_header = ['frame_number', 'person_id',
                      'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        for i in range(0, 17):
            csv_header.append('x_' + str(i))
            csv_header.append('y_' + str(i))
            csv_header.append('c_' + str(i))
        writer.writerow(csv_header)


class CSVWriter:
    """
    Threaded CSV writer to improve performance when writing keypoints data.

    This class uses a queue and worker threads to write data to a CSV file
    without blocking the main thread.
    """

    def __init__(self, filename, max_queue_size=1000, num_threads=4):
        """
        Initialize the CSV writer with a filename and worker threads.

        Parameters:
        - filename (str): Path to the CSV file to write to
        - max_queue_size (int): Maximum size of the queue before blocking
        - num_threads (int): Number of worker threads to use
        """
        self.filename = filename
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.shutdown_flag = threading.Event()
        self.num_threads = num_threads
        self.workers = []
        self._start_worker_threads()

    def _start_worker_threads(self):
        """Start worker threads for CSV writing."""
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self.workers.append(thread)

    def _worker(self):
        """Worker thread function to write data from the queue to the CSV file."""
        while not self.shutdown_flag.is_set() or not self.queue.empty():
            try:
                data = self.queue.get(timeout=0.5)
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error writing to CSV: {e}")
                self.queue.task_done()

    def write(self, data):
        """
        Queue data to be written to the CSV file.

        Parameters:
        - data (list): Row data to write to the CSV file
        """
        self.queue.put(data)

    def shutdown(self):
        """Shut down the worker threads and wait for the queue to empty."""
        self.shutdown_flag.set()
        for thread in self.workers:
            thread.join()
        logging.info("All CSV writer threads have been shut down.")


def prepare_and_write_data(frame_number, person_id, bbox, keypoints, csv_writer):
    """
    Prepare keypoint data and queue it for writing to CSV.

    Parameters:
    - frame_number (int): Frame number
    - person_id (int): Person identifier
    - bbox (dict): Bounding box coordinates
    - keypoints (dict): Keypoint coordinates and confidence
    - csv_writer (CSVWriter): CSV writer instance
    """
    csv_data = [frame_number, person_id, bbox['x1'],
                bbox['y1'], bbox['x2'], bbox['y2']]
    x = keypoints['x']
    y = keypoints['y']
    c = keypoints['visible']
    for i in range(0, 17):
        csv_data.extend([x[i], y[i], c[i]])
    csv_writer.write(csv_data)


def write_no_detections_data(frame_number, csv_writer):
    """
    Write a row of zeros to the CSV file when there are no detections in a frame.

    Parameters:
    - frame_number: The frame number where no detections were found.
    - csv_writer: The CSVWriter instance to queue the data to.

    Returns:
    - None
    """
    no_detections_data = [frame_number, 'No detections', 0, 0, 0, 0]
    for _ in range(0, 17):
        no_detections_data.extend([0, 0, 0])
    csv_writer.write(no_detections_data)


def extract_keypoints_and_save(yolo_model, video_source, csv_filename):
    """
    Extract the keypoints and bounding box coordinates from the video source using the YOLO model, track the person across frames,
    and save them to a csv file. Uses threading for faster CSV writing.

    Parameters:
    - yolo_model: The YOLO model to use for extracting the keypoints and bounding box coordinates.
    - video_source: The video source to extract the keypoints and bounding box coordinates from.
    - csv_filename: The name of the csv file to save the keypoints and bounding box coordinates to.

    Returns:
    - None
    """
    # Convert paths to Path objects for consistency
    video_source = Path(video_source)
    csv_filename = Path(csv_filename)

    # Ensure output directory exists
    csv_filename.parent.mkdir(parents=True, exist_ok=True)

    csv_writer = None

    try:
        # Create the csv file and write the header to it
        create_csv(csv_filename)

        # Initialize threaded CSV writer
        csv_writer = CSVWriter(csv_filename)

        # Load the YOLO model
        model = YOLO(yolo_model)

        # Track the keypoints and bounding box coordinates in the video source
        results = model.track(source=str(video_source), stream=True)

        if results is None:
            logging.error("No results from the YOLO model.")
            if csv_writer:
                csv_writer.shutdown()
            return

        # Iterate over each frame in the video source
        frame_number = 0
        for result in results:
            # Convert the result to json format
            try:
                json_data = json.loads(result.to_json())
            except Exception as e:
                logging.error(
                    f"Error parsing JSON data for frame {frame_number}: {e}")
                frame_number += 1
                continue

            # Check if there are any detections in the frame
            if not json_data:
                # Write no detections data to CSV
                write_no_detections_data(frame_number, csv_writer)
                frame_number += 1
                continue

            # Iterate over each person in the frame
            for person in json_data:
                if 'track_id' not in person:
                    continue

                person_id = person['track_id']
                bbox = person['box']
                keypoints = person['keypoints']

                # Prepare and queue data for CSV writing
                prepare_and_write_data(
                    frame_number, person_id, bbox, keypoints, csv_writer)

            frame_number += 1

        # Shutdown the CSV writer and wait for all data to be written
        if csv_writer:
            csv_writer.shutdown()

    except Exception as e:
        logging.error(f"An error occurred during keypoint extraction: {e}")
        # Ensure CSV writer is properly shut down in case of errors
        if csv_writer:
            try:
                csv_writer.shutdown()
            except:
                pass
