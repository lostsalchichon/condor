# radar.py
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
from speed_estimation import estimate_speed_from_video_frame

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
names = model.model.names

# Alarm path
warning_path = './alarms/warning.wav'

def play_alarm():
    subprocess.Popen(['afplay', warning_path])  # Play the sound asynchronously

def detect_objects(frame):
    h, w = frame.shape[:2]
    
    # Perform detection
    results = model(frame)

    detected_classes = set()
    for result in results:
        for detection in result.boxes:
            confidence = detection.conf.item()  # Convert tensor to Python float
            if confidence > 0.2:  # Minimum confidence to filter weak detections
                idx = int(detection.cls.item())  # Convert tensor to Python int
                class_name = model.names[idx]
                if class_name == "iha":  # Adjust to detect only drones
                    box = detection.xyxy[0].cpu().numpy()
                    (startX, startY, endX, endY) = box.astype("int")

                    label = "{}: {:.2f}%".format(class_name, confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"Detected {label} at ({startX}, {startY}), ({endX}, {endY})")
                    detected_classes.add(class_name)

    if detected_classes:
        play_alarm()

    # Integrate speed estimation
    frame = estimate_speed_from_video_frame(frame)

    return frame