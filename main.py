import cv2
import os
import yaml
import math
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# Load class names
with open("C:/Users/ashri/Desktop/desktop/helmetdetector/data/data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml["names"]

# Set target class
target_class_name = "Without Helmet"
if target_class_name not in class_names:
    raise ValueError(f"Class '{target_class_name}' not found in class list: {class_names}")
target_class_index = class_names.index(target_class_name)

# Output directory
output_dir = "detections_without_helmet_bigframe"
os.makedirs(output_dir, exist_ok=True)

# Load video
video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
save_count = 0

# Save last seen detection centers to filter duplicates
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

                # Calculate center
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Skip if already seen nearby
                if is_duplicate(cx, cy, saved_centers):
                    continue

                # Mark this center as saved
                saved_centers.append((cx, cy))

                # Add padding
                pad = 300
                height, width = frame.shape[:2]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(width, x2 + pad)
                y2 = min(height, y2 + pad)

                cropped = frame[y1:y2, x1:x2]
                save_path = os.path.join(output_dir, f"frame{frame_count}_obj{save_count}.jpg")
                cv2.imwrite(save_path, cropped)
                save_count += 1
                print(f"Saved: {class_names[class_id]} at frame {frame_count}, position ({cx},{cy})")

    frame_count += 1

cap.release()
print(f"Saved {save_count} unique '{target_class_name}' detections to '{output_dir}'")
