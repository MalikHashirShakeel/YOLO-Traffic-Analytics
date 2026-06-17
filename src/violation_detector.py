# src/violation_detector.py
import cv2
from ultralytics import YOLO
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime
import os

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 1280
TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Violation Settings
SPEED_LIMIT = 65
VIOLATION_FRAMES = 20               # Must be continuous frames to confirm violation
WRONG_WAY_THRESHOLD = -4.0

# Paths
os.makedirs("data/output/violations", exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/violation_output.mp4', fourcc, video_fps, (frame_width, frame_height))

# Tracking structures
track_history = defaultdict(list)
track_class_votes = defaultdict(list)
track_class = {}
track_start_frame = {}
track_speeds = defaultdict(list)

# Violation-specific structures
track_speed_counter = defaultdict(int)
track_wrongway_counter = defaultdict(int)
violations_logged = set()
violations_log = []
active_violations = {}

frame_count = 0


def get_pixels_per_meter(y_position, frame_height):
    ratio = y_position / frame_height
    BASE_PPM = 2.5
    MAX_PPM = 365.0
    EXPONENT = 2.4
    effective_ppm = BASE_PPM + (MAX_PPM - BASE_PPM) * (ratio ** EXPONENT)
    return max(effective_ppm, BASE_PPM)


print(f"Violation Detection Started | Speed Limit: {SPEED_LIMIT} km/h | Required Frames: {VIOLATION_FRAMES}")

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

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            if width < 25 or height < 25:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_id = int(box.id[0])
            confidence = float(box.conf[0])

            # === Exact same as speed_estimator.py ===
            track_history[track_id].append(center)
            if len(track_history[track_id]) > 25:
                track_history[track_id].pop(0)

            if track_id not in track_start_frame:
                track_start_frame[track_id] = frame_count

            # === Exact same class voting as speed_estimator.py ===
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

            # === Exact same speed estimation as speed_estimator.py ===
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

                    # === Violation Detection — only inside valid speed block ===
                    violation_type = None

                    # 1. Overspeeding — hard reset counter if not speeding
                    if avg_speed > SPEED_LIMIT:
                        track_speed_counter[track_id] += 1
                    else:
                        track_speed_counter[track_id] = 0

                    if track_speed_counter[track_id] >= VIOLATION_FRAMES:
                        violation_type = "OVERSPEEDING"

                    # 2. Wrong way — hard reset counter if direction normalizes
                    dy = history[-1][1] - history[-6][1]
                    if dy < WRONG_WAY_THRESHOLD:
                        track_wrongway_counter[track_id] += 1
                    else:
                        track_wrongway_counter[track_id] = 0

                    if track_wrongway_counter[track_id] >= VIOLATION_FRAMES:
                        violation_type = "WRONG WAY"

                    # === Log and save ONCE, only after 20 continuous frames ===
                    if violation_type:
                        active_violations[track_id] = violation_type
                        log_key = (track_id, violation_type)

                        if log_key not in violations_logged:
                            violations_logged.add(log_key)
                            violations_log.append({
                                "frame": frame_count,
                                "timestamp": f"{frame_count / video_fps:.1f}s",
                                "track_id": track_id,
                                "type": violation_type,
                                "speed": round(avg_speed, 1),
                                "class": track_class[track_id]
                            })

                            # Save cropped evidence only on first confirmed detection
                            crop = frame[max(0, y1):y2, max(0, x1):x2]
                            ts = datetime.now().strftime("%H%M%S_%f")[:10]
                            safe_type = violation_type.replace(" ", "_")
                            filename = f"data/output/violations/{safe_type}_{track_id}_{ts}.jpg"
                            cv2.imwrite(filename, crop)
                            print(f"  ⚠ [{frame_count / video_fps:.1f}s] {violation_type} — "
                                  f"{track_class[track_id]} #{track_id} @ {avg_speed:.1f} km/h")
                    else:
                        active_violations.pop(track_id, None)

                    # === Drawing ===
                    current_violation = active_violations.get(track_id)
                    if current_violation:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(annotated_frame, "!!! VIOLATION !!!", (x1, max(y1 - 35, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                        cv2.putText(annotated_frame, current_violation, (x1, max(y1 - 12, 40)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label with speed
                    label = f"{track_class[track_id]} #{track_id}  {avg_speed:.1f} km/h"
                    cv2.putText(annotated_frame, label, (x1, max(y1 - 55, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Trail
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], False, (0, 165, 255), 2)

    # HUD
    total = len(violations_log)
    if total > 0:
        cv2.putText(annotated_frame, f"VIOLATIONS: {total}",
                    (20, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    out.write(annotated_frame)
    cv2.imshow("Violation Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Final Report
print("\n✅ Violation Detection Completed!")
print(f"Total Unique Violations: {len(violations_log)}")
if violations_log:
    print("\nAll Violations:")
    for v in violations_log:
        print(f"  • {v['timestamp']:>8} | {v['type']:>15} | "
              f"{v['class']} #{v['track_id']} @ {v['speed']} km/h")

print(f"\nViolation crops saved in: data/output/violations/")