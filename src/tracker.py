# src/tracking.py
import cv2
from ultralytics import YOLO
from collections import defaultdict

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 1280

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/tracking_output.mp4', fourcc, 30.0, (frame_width, frame_height))

track_history = defaultdict(lambda: [])
vehicle_counts = defaultdict(int)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # === Tracking with YOLOv8m ===
    results = model.track(
        frame,
        persist=True,              # Critical for maintaining IDs
        tracker="botsort.yaml",    # Best for traffic (you can try "bytetrack.yaml")
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:   # Check if tracks exist
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Store trajectory
            track_history[track_id].append((center_x, center_y))

            # Draw box + ID
            label = f"{VEHICLE_CLASSES[cls]} #{track_id}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Optional: Draw trajectory trail
            if len(track_history[track_id]) > 1:
                points = track_history[track_id][-30:]  # last 30 points
                for i in range(1, len(points)):
                    cv2.line(annotated_frame, points[i-1], points[i], (0, 165, 255), 2)

    out.write(annotated_frame)
    cv2.imshow("YOLOv8m + BoT-SORT Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Tracking completed. Output saved to data/output/tracking_output.mp4")