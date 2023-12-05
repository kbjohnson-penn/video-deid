from ultralytics import YOLO
import numpy as np
import sys


def predict_pose(yolo_model, video_source, keypoints_path, stream=False):
    # Load a pretrained YOLOv8n model
    model = YOLO(yolo_model)

    # Run inference on the source
    results = model(video_source, stream=stream, show_labels=True, save_txt=True,
                    save_frames=True, save=True, save_conf=True)

    # Open the keypoints file
    with open(keypoints_path, 'w') as f:
        # Process results
        for result in results:
            # Extracting keypoints
            keypoints = result.keypoints  # Keypoints object for pose outputs
            if keypoints.xy is not None:
                # Convert the tensor to a numpy array and then to a list
                keypoints_list = keypoints.xy.cpu().numpy().tolist()
                # Convert list to string and write to file
                keypoints_str = ', '.join([str(kp) for kp in keypoints_list])
                f.write(keypoints_str + '\n')


def main():
    yolo_model = 'yolov8m-pose.pt'
    video_source = sys.argv[1]
    keypoints_path = sys.argv[2]
    stream_mode = True  # or False, depending on your needs
    predict_pose(yolo_model, video_source, keypoints_path, stream=stream_mode)


if __name__ == "__main__":
    # Ensure the correct number of arguments are provided
    num_args = len(sys.argv) - 1
    if num_args != 2:
        print("Usage: python pose-detection.py <video_source> <keypoints_path>")
        sys.exit(1)

    main()
    sys.exit(0)
