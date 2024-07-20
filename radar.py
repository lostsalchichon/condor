import cv2
import numpy as np

# Load the COCO class labels the model was trained on
CLASSES = ["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "person", "bicycle", "car", 
           "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", 
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
           "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", 
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", 
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", 
           "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
           "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
           "toothbrush"]

# Load the serialized model from disk
prototxt_path = './data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model_path = './data/frozen_inference_graph.pb'
net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)

def detect_objects(frame):
    h, w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (300, 300))
    
    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (300, 300), (0, 0, 0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Minimum confidence to filter weak detections
            idx = int(detections[0, 0, i, 1])
            if idx < len(CLASSES):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Detected {label} at ({startX}, {startY}), ({endX}, {endY})")

    return frame