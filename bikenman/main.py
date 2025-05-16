import cv2
import os
import math
import yaml
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Load class names
with open("C:/Users/ashri/Desktop/desktop/helmetdetector/bikenman/data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml["names"]

# Target class (update this based on the class name in your YAML)
target_class_name = "Without Helmet"  # replace with your actual class name if different

# Get index of target class
if target_class_name not in class_names:
    raise ValueError(f"Class '{target_class_name}' not found. Available classes: {class_names}")
target_class_index = class_names.index(target_class_name)

# Create output directory
output_dir = "detections_without_helmet"
os.makedirs(output_dir, exist_ok=True)

# Load the video
video_path = "traffic.mp4"  # Update with your video filename
cap = cv2.VideoCapture(video_path)

frame_count = 0
save_count = 0
saved_centers = []

def is_duplicate(cx, cy, saved_centers, threshold=100):
    for sx, sy in saved_centers:
        if math.hypot(cx - sx, cy - sy) < threshold:
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == target_class_index:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if is_duplicate(cx, cy, saved_centers):
                    continue

                saved_centers.append((cx, cy))

                # Add padding to crop
                pad = 50
                height, width = frame.shape[:2]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(width, x2 + pad)
                y2 = min(height, y2 + pad)

                cropped = frame[y1:y2, x1:x2]
                save_path = os.path.join(output_dir, f"frame{frame_count}_obj{save_count}.jpg")
                cv2.imwrite(save_path, cropped)
                print(f"Saved: {target_class_name} at frame {frame_count}, position ({cx},{cy})")
                save_count += 1

    frame_count += 1

cap.release()
print(f"Saved {save_count} detections of class '{target_class_name}' to '{output_dir}'")
