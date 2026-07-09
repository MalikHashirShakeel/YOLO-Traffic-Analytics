# src/speed_estimator.py
import cv2
from ultralytics import YOLO
from collections import defaultdict, Counter

from utils import (
    parse_box, update_class, estimate_speed,
    check_line_cross, draw_box_label, draw_trail, draw_speed
)

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH  = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 1280
TRACKER  = "bytetrack.yaml"
VIDEO_FPS = 25.0

model = YOLO(MODEL_PATH)

cap          = cv2.VideoCapture(VIDEO_PATH)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps    = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/speed_output.mp4', fourcc, video_fps,
                      (frame_width, frame_height))

# Tracking structures
track_history     = defaultdict(list)
track_class_votes = defaultdict(list)
track_class       = {}
track_start_frame = {}
track_speeds      = defaultdict(list)
vehicle_counter   = Counter()
track_crossed     = set()

frame_count = 0
LINE_Y = int(frame_height * 0.65)

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
    cv2.line(annotated_frame, (50, LINE_Y), (frame_width - 50, LINE_Y),
             (0, 255, 255), 3)

    if results[0].boxes.id is not None:
        for box in results[0].boxes:

            # 1. Detection — parse & filter
            b = parse_box(box)
            if b is None:
                continue

            track_id = b["track_id"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

            # Update position history
            track_history[track_id].append(b["center"])
            if len(track_history[track_id]) > 25:
                track_history[track_id].pop(0)

            if track_id not in track_start_frame:
                track_start_frame[track_id] = frame_count

            # 2. Class voting
            update_class(track_id, b["cls_id"], b["confidence"],
                         track_class_votes, track_class)

            age = frame_count - track_start_frame[track_id]
            if age < 10:
                continue

            # 3. Speed estimation
            avg_speed = estimate_speed(
                track_id, track_history, track_speeds, video_fps, frame_height
            )
            if avg_speed is not None:
                draw_speed(annotated_frame, avg_speed, x1, y2)

            # 4. Counting
            check_line_cross(track_id, track_history, track_class,
                             track_crossed, vehicle_counter, LINE_Y)

            # 5. Visuals
            draw_box_label(annotated_frame, x1, y1, x2, y2,
                           f"{track_class[track_id]} #{track_id}")
            draw_trail(annotated_frame, track_history, track_id)

    # Stats HUD
    stats     = f"Frame: {frame_count} | Counts: {dict(vehicle_counter)}"
    text_size = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    cv2.rectangle(annotated_frame, (8, 8), (15 + text_size[0], 45), (0, 0, 0), -1)
    cv2.putText(annotated_frame, stats, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    out.write(annotated_frame)
    cv2.imshow("Speed Estimation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Speed Estimation completed!")
print("Final Counts:", dict(vehicle_counter))