# radar.py
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
from speed_estimation import estimate_speed_from_video_frame
from obj_crop import crop_objects
from objexclude.birdexclude import bird_exclude, crop_birds

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
names = model.model.names

# Alarm path
warning_path = './alarms/warning.wav'

# Speech path
speech_path = './speech/detected.wav'

# Flag to check if speech has been played
speech_played = False

def play_speech():
    subprocess.Popen(['afplay', speech_path])  # Play speech asynchronously

def play_alarm():
    subprocess.Popen(['afplay', warning_path])  # Play the sound asynchronously

def detect_objects(frame):
    global speech_played  # Declare speech_played as global
    h, w = frame.shape[:2]

    # Perform drone detection
    results = model(frame)
    detected_classes = set()

    # Perform bird detection
    bird_detections = bird_exclude(frame)
    bird_boxes = {tuple(box): confidence for box, confidence, _ in bird_detections}

    for result in results:
        for detection in result.boxes:
            confidence = detection.conf.item()
            if confidence > 0.4:
                idx = int(detection.cls.item())
                class_name = model.names[idx]
                box = tuple(detection.xyxy[0].cpu().numpy().astype("int"))

                if class_name == "iha":
                    bird_confidence = bird_boxes.get(box)
                    if bird_confidence and bird_confidence > confidence:
                        crop_birds(frame, [(box, bird_confidence, class_name)])
                    else:
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence * 100:.2f}%"
                        y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                        cv2.putText(frame, label, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_classes.add(class_name)
                        crop_objects(frame, [result])

    if detected_classes:
        if not speech_played:
            play_speech()
            speech_played = True
        play_alarm()
    else:
        speech_played = False  # Reset the flag if no classes are detected

    # Integrate speed estimation
    frame = estimate_speed_from_video_frame(frame)

    return frame