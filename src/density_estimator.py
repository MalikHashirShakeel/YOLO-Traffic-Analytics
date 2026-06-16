# src/density_estimator.py
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 1280
TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Detection filters (from speed_estimator.py)
MIN_BOX_WIDTH = 25
MIN_BOX_HEIGHT = 25

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/density_output.mp4', fourcc, video_fps, (frame_width, frame_height))

# Tracking structures
track_history = defaultdict(list)
track_class_votes = defaultdict(list)   # (cls_id, confidence) history per track
track_class = {}                        # resolved class per track
track_start_frame = {}
frame_count = 0

# === Expanded Density Zone ===
DENSITY_ZONE = np.array([
    [int(frame_width * 0.10), int(frame_height * 0.25)],   # Top-left (expanded)
    [int(frame_width * 0.88), int(frame_height * 0.25)],   # Top-right
    [int(frame_width * 0.92), int(frame_height * 0.72)],   # Bottom-right
    [int(frame_width * 0.08), int(frame_height * 0.72)],   # Bottom-left
], np.int32)

# Density Thresholds
DENSITY_THRESHOLDS = {
    "LOW": 4,
    "MEDIUM": 5,
    "HIGH": 13,
}

def get_density_level(vehicle_count: int):
    if vehicle_count <= DENSITY_THRESHOLDS["LOW"]:
        return "LOW", (0, 255, 0), "Light Traffic"
    elif vehicle_count <= DENSITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM", (0, 255, 255), "Moderate Traffic"
    elif vehicle_count <= DENSITY_THRESHOLDS["HIGH"]:
        return "HIGH", (0, 165, 255), "Heavy Traffic"
    else:
        return "JAM", (0, 0, 255), "CONGESTION / JAM"


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.5,
        verbose=False
    )

    annotated_frame = frame.copy()
    active_vehicles_in_zone = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            # === Same detection logic as speed_estimator.py ===
            if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_id = int(box.id[0])
            confidence = float(box.conf[0])

            # Update history
            track_history[track_id].append(center)
            if len(track_history[track_id]) > 25:
                track_history[track_id].pop(0)

            if track_id not in track_start_frame:
                track_start_frame[track_id] = frame_count

            # === Confidence-weighted majority voting for stable class ===
            track_class_votes[track_id].append((cls_id, confidence))
            if len(track_class_votes[track_id]) > 15:
                track_class_votes[track_id].pop(0)

            class_scores = defaultdict(float)
            for cid, conf in track_class_votes[track_id]:
                class_scores[cid] += conf
            best_cls = max(class_scores, key=class_scores.get)
            track_class[track_id] = VEHICLE_CLASSES[best_cls]

            # Count mature tracks in zone
            if frame_count - track_start_frame[track_id] >= 8:
                if cv2.pointPolygonTest(DENSITY_ZONE, center, False) >= 0:
                    active_vehicles_in_zone += 1

            # Draw detections
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{track_class[track_id]} #{track_id}"
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Trail
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], False, (0, 165, 255), 2)

    # === Density Analysis ===
    density_level, color, description = get_density_level(active_vehicles_in_zone)

    # Draw Density Zone
    cv2.polylines(annotated_frame, [DENSITY_ZONE], True, (255, 255, 100), 4)

    # Density Info Panel
    panel_x = 15
    cv2.rectangle(annotated_frame, (panel_x-5, 10), (panel_x + 380, 190), (0, 0, 0), -1)

    texts = [
        f"Frame: {frame_count}",
        f"Vehicles in Zone: {active_vehicles_in_zone}",
        f"Density Level: {density_level}",
        f"Status: {description}"
    ]

    for i, text in enumerate(texts):
        cv2.putText(annotated_frame, text, (panel_x, 40 + i*35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Large Status Indicator
    cv2.rectangle(annotated_frame, (frame_width-320, 25), (frame_width-20, 110), (0, 0, 0), -1)
    cv2.putText(annotated_frame, "TRAFFIC DENSITY", (frame_width-305, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated_frame, density_level, (frame_width-240, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 4)

    # Progress Bar
    bar_width = 280
    fill = int(bar_width * min(active_vehicles_in_zone / 15, 1))
    cv2.rectangle(annotated_frame, (frame_width-300, 125), (frame_width-20, 155), (60, 60, 60), -1)
    cv2.rectangle(annotated_frame, (frame_width-300, 125), (frame_width-300 + fill, 155), color, -1)

    out.write(annotated_frame)
    cv2.imshow("Traffic Density Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Density Estimation completed!")