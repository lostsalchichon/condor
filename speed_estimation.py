# speed_estimation.py
import cv2
from ultralytics import YOLO, solutions

# Load the YOLO model
model = YOLO('yolov8s.pt')
names = model.model.names

# Initialize the SpeedEstimator
line_pts = [(0, 360), (1280, 360)]
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=False,
)

def estimate_speed_from_video_frame(frame):
    tracks = model.track(frame, persist=True, show=False)
    frame_with_speed = speed_obj.estimate_speed(frame, tracks)
    return frame_with_speed
