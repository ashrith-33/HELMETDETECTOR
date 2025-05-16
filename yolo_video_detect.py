from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt, yolov8m.pt, etc.

# Load your input video
video_path = "traffic.mp4"  # Replace with your actual video file
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up the output video writer for MP4 format
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("🚀 Processing video...")

frame_number = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 detection on the current frame
    results = model(frame, stream=True)

    # Annotate and write the frame
    for r in results:
        annotated_frame = r.plot()
        out.write(annotated_frame)

    frame_number += 1
    print(f"Processing frame {frame_number}/{total_frames}", end="\r")

# Release resources
cap.release()
out.release()

print("\n✅ Done! Output video saved as 'output.mp4'")
