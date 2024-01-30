from ultralytics import YOLO


def save_results(results, keypoints_path):
    """
    Save the results of the pose detection model to a file.

    Parameters:
    results: Results of the pose detection model.
    keypoints_path: Path to the keypoints file.

    Returns:
    None
    """

    # Open the keypoints file
    with open(keypoints_path, 'w') as f:
        # Process results
        for result in results:
            # Extracting keypoints
            keypoints = result.keypoints  # Keypoints object for pose outputs
            if keypoints.xyn is not None and keypoints.conf is not None:
                # Convert the tensor to a numpy array and then to a list
                keypoints_xyn_list = keypoints.xyn.cpu().numpy().tolist()
                keypoints_conf_list = keypoints.conf.cpu().numpy().tolist()

                keypoints_with_conf = []
                for (person_keypoints, confidence_socres) in zip(keypoints_xyn_list, keypoints_conf_list):
                    key_list = []
                    for (x, y), conf in zip(person_keypoints, confidence_socres):
                        key_list.append([x, y, conf])
                    keypoints_with_conf.append(key_list)

                # Convert list to string and write to file
                keypoints_str = ', '.join([str(kp)
                                          for kp in keypoints_with_conf])
                f.write(keypoints_str + '\n')


def predict_pose(yolov8_pose_model, video_source):
    """
    Predict the pose of the people in the video.

    Parameters:
    yolov8_pose_model: Path to the YOLOv8n pose model.
    video_source: Path to the video source.

    Returns:
    Results of the pose detection model.
    """

    # Load a pretrained YOLOv8n model
    model = YOLO(yolov8_pose_model)

    # Run inference on the source
    results = model.track(source=video_source,
                          stream=True, tracker="botsort.yaml")

    return results


def predict_pose_and_save_results(video_source, keypoints_path):
    """
    Predict the pose of the people in the video and save the results to a file.

    Parameters:
    video_source: Path to the video source.
    keypoints_path: Path to the keypoints file.

    Returns:
    None
    """

    yolov8_pose_model = 'yolov8x-pose-p6.pt'
    prediction_results = predict_pose(
        yolov8_pose_model=yolov8_pose_model, video_source=video_source)
    save_results(prediction_results, keypoints_path)
