import streamlit as st
import cv2
import numpy as np
from radar import detect_objects

# Set page configuration
st.set_page_config(page_title="Condor - Radar de UAVs", page_icon="✈", layout="wide")

# Custom CSS for black minimalistic background
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .css-18e3th9 {
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("Condor - Radar de UAVs")

# Function to get camera feed
def get_camera_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return np.zeros((480, 640, 3), dtype=np.uint8)

# Stream camera feed
frame = get_camera_frame()
if frame is not None:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_detections = detect_objects(frame)
    st.image(frame_with_detections, channels="RGB")

# Run the Streamlit app
if __name__ == "__main__":
    st.rerun()