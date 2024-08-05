# birdexclude.py
import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load the bird detection model
model_path = os.path.join(os.path.dirname(__file__), 'birdmodel.pt')
model = YOLO(model_path)
names = model.model.names

# Directory to save cropped bird objects
crop_dir_name = "./objexclude/birds"
if not os.path.exists(crop_dir_name):
    os.makedirs(crop_dir_name)

def bird_exclude(frame):
    """
    Detect birds in the frame and return their bounding boxes and confidence levels.
    """
    results = model(frame)
    bird_detections = []

    for result in results:
        for detection in result.boxes:
            confidence = detection.conf.item()
            if confidence > 0.4:
                idx = int(detection.cls.item())
                class_name = model.names[idx]
                if class_name == "bird":
                    box = detection.xyxy[0].cpu().numpy().astype("int")
                    bird_detections.append((box, confidence, class_name))

    return bird_detections

def crop_birds(frame, bird_detections):
    """
    Crop detected birds from the frame and save them to the specified directory.
    """
    idx = 0
    for box, confidence, class_name in bird_detections:
        idx += 1
        crop_obj = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cv2.imwrite(os.path.join(crop_dir_name, f"{idx}.png"), crop_obj)