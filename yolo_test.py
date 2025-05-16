from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model (YOLOv8n = nano, small size & fast)
model = YOLO('yolov8n.pt')  # It will auto-download this on first use

# Load an image or a video frame
img = cv2.imread('test.jpg')  # Replace with your image name or path

# Run YOLOv8 object detection
results = model(img)

# Visualize the results on the image
annotated_frame = results[0].plot()

import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.title("YOLO Detection")
plt.axis("off")
plt.show()