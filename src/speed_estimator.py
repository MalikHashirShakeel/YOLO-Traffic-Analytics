# src/speed_estimator.py
import cv2
from ultralytics import YOLO
from collections import defaultdict, Counter
import numpy as np

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 1280
TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Speed settings
VIDEO_FPS = 25.0
MIN_BOX_WIDTH = 25
MIN_BOX_HEIGHT = 25

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/speed_output.mp4', fourcc, video_fps, (frame_width, frame_height))

# Tracking structures
track_history = defaultdict(list)
track_class_votes = defaultdict(list)  # (cls_id, confidence) history per track
track_class = {}                        # resolved class per track
track_start_frame = {}
track_speeds = defaultdict(list)
vehicle_counter = Counter()
track_crossed = set()

frame_count = 0
LINE_Y = int(frame_height * 0.65)


def get_pixels_per_meter(y_position, frame_height):
    ratio = y_position / frame_height
    BASE_PPM = 2.5
    MAX_PPM = 365.0
    EXPONENT = 2.4
    effective_ppm = BASE_PPM + (MAX_PPM - BASE_PPM) * (ratio ** EXPONENT)
    return max(effective_ppm, BASE_PPM)


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
    cv2.line(annotated_frame, (50, LINE_Y), (frame_width - 50, LINE_Y), (0, 255, 255), 3)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_id = int(box.id[0])
            confidence = float(box.conf[0])

            track_history[track_id].append(center)
            if len(track_history[track_id]) > 25:
                track_history[track_id].pop(0)

            if track_id not in track_start_frame:
                track_start_frame[track_id] = frame_count

            # === Confidence-weighted majority voting for class ===
            track_class_votes[track_id].append((cls_id, confidence))
            if len(track_class_votes[track_id]) > 15:
                track_class_votes[track_id].pop(0)

            class_scores = defaultdict(float)
            for cid, conf in track_class_votes[track_id]:
                class_scores[cid] += conf
            best_cls = max(class_scores, key=class_scores.get)
            track_class[track_id] = VEHICLE_CLASSES[best_cls]

            age = frame_count - track_start_frame[track_id]
            if age < 10:
                continue

            # === Speed Estimation ===
            history = track_history[track_id]
            if len(history) >= 6:
                recent_points = np.array(history[-6:])
                pixel_distance = np.linalg.norm(recent_points[-1] - recent_points[0])

                frames_elapsed = len(recent_points) - 1
                time_elapsed = frames_elapsed / video_fps

                if time_elapsed > 0:
                    avg_y = np.mean(recent_points[:, 1])
                    effective_ppm = get_pixels_per_meter(avg_y, frame_height)

                    speed_mps = pixel_distance / time_elapsed / effective_ppm
                    speed_kmh = speed_mps * 3.6

                    if pixel_distance < 2.5:
                        speed_kmh = 0.0
                    else:
                        speed_kmh = max(3.0, min(speed_kmh, 130))

                    track_speeds[track_id].append(speed_kmh)
                    if len(track_speeds[track_id]) > 10:
                        track_speeds[track_id].pop(0)

                    avg_speed = np.mean(track_speeds[track_id])

                    speed_text = f"{avg_speed:.1f} km/h"
                    color = (0, 255, 0) if avg_speed > 8 else (0, 180, 255)
                    cv2.putText(annotated_frame, speed_text, (x1, y2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Counting
            if track_id not in track_crossed:
                history_y = [p[1] for p in track_history[track_id]]
                if len(history_y) >= 2 and history_y[-2] < LINE_Y <= history_y[-1]:
                    vehicle_counter[track_class[track_id]] += 1
                    track_crossed.add(track_id)

            # Visuals
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{track_class[track_id]} #{track_id}"
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], False, (0, 165, 255), 2)

    # Stats
    stats = f"Frame: {frame_count} | Counts: {dict(vehicle_counter)}"
    text_size = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    cv2.rectangle(annotated_frame, (8, 8), (15 + text_size[0], 45), (0, 0, 0), -1)
    cv2.putText(annotated_frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    out.write(annotated_frame)
    cv2.imshow("Speed Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Speed Estimation completed!")
print("Final Counts:", dict(vehicle_counter))