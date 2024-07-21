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

# Function to get camera feed continuously
def get_camera_feed():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break
    cap.release()

# Display the video feed in real-time
frame_generator = get_camera_feed()

# Create a placeholder to continuously update the frame
frame_placeholder = st.empty()

try:
    for frame in frame_generator:
        frame_with_detections = detect_objects(frame)
        frame_placeholder.image(cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB), channels="RGB")
except RuntimeError as e:
    st.error(f"Unable to access the webcam: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    st.rerun()
