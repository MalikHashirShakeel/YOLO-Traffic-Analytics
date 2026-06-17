# src/density_estimator.py
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

from utils import (
    parse_box, update_class,
    is_in_zone, get_density_level, draw_density_panel,
    draw_box_label, draw_trail
)

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 1280
TRACKER = "bytetrack.yaml"

model = YOLO(MODEL_PATH)

cap          = cv2.VideoCapture(VIDEO_PATH)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/density_output.mp4', fourcc, video_fps,
                      (frame_width, frame_height))

# Tracking structures
track_history     = defaultdict(list)
track_class_votes = defaultdict(list)
track_class       = {}
track_start_frame = {}
frame_count       = 0

# === Density Zone ===
DENSITY_ZONE = np.array([
    [int(frame_width * 0.10), int(frame_height * 0.25)],
    [int(frame_width * 0.88), int(frame_height * 0.25)],
    [int(frame_width * 0.92), int(frame_height * 0.72)],
    [int(frame_width * 0.08), int(frame_height * 0.72)],
], np.int32)

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

            # 3. Zone count — mature tracks only
            if frame_count - track_start_frame[track_id] >= 8:
                if is_in_zone(b["center"], DENSITY_ZONE):
                    active_vehicles_in_zone += 1

            # 4. Visuals
            draw_box_label(annotated_frame, x1, y1, x2, y2,
                           f"{track_class[track_id]} #{track_id}")
            draw_trail(annotated_frame, track_history, track_id)

    # 5. Density analysis & drawing
    density_level, color, description = get_density_level(active_vehicles_in_zone)

    cv2.polylines(annotated_frame, [DENSITY_ZONE], True, (255, 255, 100), 4)

    draw_density_panel(annotated_frame, frame_count, active_vehicles_in_zone,
                       density_level, description, color,
                       frame_width, frame_height)

    out.write(annotated_frame)
    cv2.imshow("Traffic Density Estimation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Density Estimation completed!")