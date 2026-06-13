# src/detector.py

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "data/input/traffic.mp4"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated_frame = frame.copy()

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            label = f"{VEHICLE_CLASSES[cls]} {confidence:.2f}"

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("Traffic Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()