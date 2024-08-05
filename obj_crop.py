# obj_cropping.py
import os
import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")
names = model.names

# Directory to save cropped objects
crop_dir_name = "./objscrop"
if not os.path.exists(crop_dir_name):
    os.makedirs(crop_dir_name)

def crop_objects(frame, results):
    """
    Crop detected objects from the frame and save them to the specified directory.
    
    Parameters:
    - frame: The current video frame.
    - results: The detection results from the YOLO model.
    """
    idx = 0
    for result in results:
        boxes = result.boxes.xyxy.cpu().tolist()
        clss = result.boxes.cls.cpu().tolist()
        annotator = Annotator(frame, line_width=2, example=names)

        if boxes is not None:
            for box, cls in zip(boxes, clss):
                idx += 1
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                crop_obj = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                timestamp = int(time.time())  # Get current timestamp
                filename = f"{timestamp}_{idx}.png"
                cv2.imwrite(os.path.join(crop_dir_name, filename), crop_obj)