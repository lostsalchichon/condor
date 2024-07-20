import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load the COCO class labels the model was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Load the serialized model from disk
net = cv2.dnn.readNetFromCaffe("path_to_deploy.prototxt", "path_to_MobileNetSSD_deploy.caffemodel")

# Initialize the webcam
cap = cv2.VideoCapture(0)

def detect_objects():
    # Grab the frame from the video stream
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image from webcam.")
        return

    # Resize the frame to have a width of 300 pixels (required by the model)
    frame_resized = cv2.resize(frame, (300, 300))

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Minimum confidence to filter weak detections
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["bird", "kite", "frisbee", "remote", "knife", "airplane"]:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        root.quit()

    # Call detect_objects again after a delay
    root.after(10, detect_objects)

# Create the main window
root = tk.Tk()
root.title("Condor - Radar de UAVs")

# Create and pack the start button
start_button = tk.Button(root, text="Start Detection", command=detect_objects)
start_button.pack()

# Run the Tkinter main loop
root.mainloop()
