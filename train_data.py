from ultralytics import YOLO

def main():
    # Load pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")

    print("Starting YOLOv8 training...")
    model.train(
        data="C:/Users/ashri/Desktop/desktop/helmetdetector/data/data.yaml",
        epochs=50,
        imgsz=640
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, but safe
    main()
