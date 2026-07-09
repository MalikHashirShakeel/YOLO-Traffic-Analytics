# src/tracking.py
import cv2
from ultralytics import YOLO
from collections import defaultdict, Counter
import numpy as np

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.37        # Slightly lowered for better distant detection
IMG_SIZE = 1280
TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/tracking_output.mp4', fourcc, fps, (frame_width, frame_height))

# Tracking structures
track_history = defaultdict(list)
track_class = {}
track_start_frame = {}
track_crossed = set()
frame_count = 0
vehicle_counter = Counter()

# Virtual counting line
LINE_Y = int(frame_height * 0.65)
LINE_COLOR = (0, 255, 255)
MIN_TRACK_AGE = 12                 # Increased for more consistency
MAX_MISSING_FRAMES = 25
MAX_TRACK_POINTS = 45

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
        iou=0.5,                   # Slightly higher IoU for stability
        verbose=False
    )

    annotated_frame = frame.copy()

    # Draw virtual counting line
    cv2.line(annotated_frame, (50, LINE_Y), (frame_width - 50, LINE_Y), LINE_COLOR, 3)

    active_ids = set()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            active_ids.add(track_id)

            # Update history
            track_history[track_id].append((center_x, center_y))
            if len(track_history[track_id]) > MAX_TRACK_POINTS:
                track_history[track_id].pop(0)

            if track_id not in track_start_frame:
                track_start_frame[track_id] = frame_count

            if track_id not in track_class:
                track_class[track_id] = VEHICLE_CLASSES[cls_id]

            age = frame_count - track_start_frame[track_id]

            # Skip immature tracks
            if age < MIN_TRACK_AGE:
                continue

            # Strict line crossing count
            crossed = False
            if track_id not in track_crossed:
                history = track_history[track_id]
                if len(history) >= 2:
                    prev_y = history[-2][1]
                    curr_y = history[-1][1]
                    if prev_y < LINE_Y and curr_y >= LINE_Y:
                        vehicle_counter[track_class[track_id]] += 1
                        track_crossed.add(track_id)
                        crossed = True

            # Visuals
            color = (0, 255, 0) if not crossed else (0, 255, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            label = f"{track_class[track_id]} #{track_id}"
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Trail
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], False, (0, 165, 255), 2)

    # Cleanup
    for tid in list(track_history.keys()):
        if frame_count - track_start_frame.get(tid, 0) > 400 or \
           (tid not in active_ids and frame_count - track_start_frame.get(tid, 0) > MAX_MISSING_FRAMES):
            track_history.pop(tid, None)
            track_class.pop(tid, None)
            track_start_frame.pop(tid, None)

    # === Improved Stats Overlay with Black Background ===
    stats = f"Frame: {frame_count} | Active: {len(active_ids)} | Counts: {dict(vehicle_counter)}"
    
    # Black background for better visibility
    text_size = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    cv2.rectangle(annotated_frame, (8, 8), (10 + text_size[0], 45), (0, 0, 0), -1)
    
    # White text on black background
    cv2.putText(annotated_frame, stats, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    out.write(annotated_frame)
    cv2.imshow("Traffic Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Tracking completed!")
print("Final Counts:", dict(vehicle_counter))