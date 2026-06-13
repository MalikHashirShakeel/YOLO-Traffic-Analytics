# src/detection.py
import cv2
from ultralytics import YOLO

# === Configuration ===
MODEL_PATH = "yolov8m.pt"          # Upgraded from yolov8s
VIDEO_PATH = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35        # Lowered for distant vehicles
IMG_SIZE = 1280                    # Higher resolution for small/distant objects
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Load model (same as your yolo_test.py)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/detection_output.mp4', fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run detection with improved settings
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            # Filter tiny boxes
            if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
                continue

            label = f"{VEHICLE_CLASSES[cls]} {confidence:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write frame to output video
    out.write(annotated_frame)
    cv2.imshow("Traffic Detection - YOLOv8m", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Detection completed. Output saved to data/output/detection_output.mp4")