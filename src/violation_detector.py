# src/violation_detector.py
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from datetime import datetime
import os

from utils import (
    parse_box, update_class, estimate_speed,
    draw_trail
)

# === Configuration ===
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH  = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 1280
TRACKER  = "bytetrack.yaml"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# === Violation Settings ===
SPEED_LIMIT         = 60       # km/h

# Continuity — must be violated for this many consecutive frames
VIOLATION_FRAMES    = 20

# Stability — across the violation window, std dev of speed must be low
# (rules out random spikes that happen to average over the limit)
SPEED_STD_MAX       = 10.0     # km/h — if std dev is higher, it's a glitch

# How far over the limit the average must be (not just barely over)
SPEED_MARGIN        = 3.0      # km/h — avg must exceed SPEED_LIMIT + SPEED_MARGIN

# Minimum frames a track must exist before any violation is even considered
MIN_TRACK_AGE       = 30       # frames

# Wrong way — how many of the last N frames must show negative dy
WRONG_WAY_THRESHOLD  = -4.0
WRONG_WAY_CONFIRM_N  = 8       # look back this many recent history points
WRONG_WAY_MIN_RATIO  = 0.75    # at least 75% of those must show wrong-way dy

os.makedirs("data/output/violations", exist_ok=True)

model = YOLO(MODEL_PATH)

cap          = cv2.VideoCapture(VIDEO_PATH)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/output/violation_output.mp4', fourcc, video_fps,
                      (frame_width, frame_height))

# Tracking structures
track_history        = defaultdict(list)
track_class_votes    = defaultdict(list)
track_class          = {}
track_start_frame    = {}
track_speeds         = defaultdict(list)

# Violation structures
track_speed_counter    = defaultdict(int)
track_wrongway_counter = defaultdict(int)

# Speed sample window for stability check — last N speed readings per track
track_speed_window     = defaultdict(list)
SPEED_WINDOW_SIZE      = 20    # must match VIOLATION_FRAMES for meaningful std dev

violations_logged      = set()
violations_log         = []
active_violations      = {}

frame_count = 0


def is_overspeeding_verified(track_id, avg_speed,
                              track_speed_counter, track_speed_window):
    """
    Returns True only if:
      1. Counter has been sustained for VIOLATION_FRAMES consecutive frames
      2. Average speed is over SPEED_LIMIT + SPEED_MARGIN (not just barely over)
      3. Speed window std dev is low — consistent, not a spike
    """
    if track_speed_counter[track_id] < VIOLATION_FRAMES:
        return False

    window = track_speed_window[track_id]
    if len(window) < VIOLATION_FRAMES:
        return False

    recent = window[-VIOLATION_FRAMES:]
    mean_speed = np.mean(recent)
    std_speed  = np.std(recent)

    if mean_speed < SPEED_LIMIT + SPEED_MARGIN:
        return False                  # barely over — not confident enough

    if std_speed > SPEED_STD_MAX:
        return False                  # too erratic — likely a glitch

    return True


def is_wrong_way_verified(track_id, history, track_wrongway_counter):
    """
    Returns True only if:
      1. Counter has been sustained for VIOLATION_FRAMES consecutive frames
      2. Across the last WRONG_WAY_CONFIRM_N history points,
         at least WRONG_WAY_MIN_RATIO fraction show negative dy — consistent direction
    """
    if track_wrongway_counter[track_id] < VIOLATION_FRAMES:
        return False

    if len(history) < WRONG_WAY_CONFIRM_N + 1:
        return False

    # Check what fraction of recent inter-frame dy values are wrong-way
    recent = history[-(WRONG_WAY_CONFIRM_N + 1):]
    wrong_way_count = sum(
        1 for i in range(1, len(recent))
        if recent[i][1] - recent[i - 1][1] < WRONG_WAY_THRESHOLD
    )
    ratio = wrong_way_count / WRONG_WAY_CONFIRM_N

    return ratio >= WRONG_WAY_MIN_RATIO


print(f"Violation Detection Started | Speed Limit: {SPEED_LIMIT} km/h | "
      f"Required Frames: {VIOLATION_FRAMES} | "
      f"Speed Margin: +{SPEED_MARGIN} | Max Std Dev: {SPEED_STD_MAX}")

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

            # Gate 1 — track must be old enough to be reliable
            if age < MIN_TRACK_AGE:
                continue

            # 3. Speed estimation — exact same function as speed_estimator.py
            avg_speed = estimate_speed(
                track_id, track_history, track_speeds, video_fps, frame_height
            )
            if avg_speed is None:
                continue

            # Accumulate speed window for stability check
            track_speed_window[track_id].append(avg_speed)
            if len(track_speed_window[track_id]) > SPEED_WINDOW_SIZE:
                track_speed_window[track_id].pop(0)

            # 4. Violation counters — increment / hard reset
            history = track_history[track_id]
            violation_type = None

            # Overspeeding counter
            if avg_speed > SPEED_LIMIT:
                track_speed_counter[track_id] += 1
            else:
                track_speed_counter[track_id] = 0

            # Wrong way counter
            if len(history) >= 6:
                dy = history[-1][1] - history[-6][1]
                if dy < WRONG_WAY_THRESHOLD:
                    track_wrongway_counter[track_id] += 1
                else:
                    track_wrongway_counter[track_id] = 0

            # 5. Verification — only confirm if all checks pass
            if is_overspeeding_verified(track_id, avg_speed,
                                        track_speed_counter,
                                        track_speed_window):
                violation_type = "OVERSPEEDING"

            if is_wrong_way_verified(track_id, history,
                                     track_wrongway_counter):
                violation_type = "WRONG WAY"

            # 6. Log and save — only once per (track_id, type), only if verified
            if violation_type:
                active_violations[track_id] = violation_type
                log_key = (track_id, violation_type)

                if log_key not in violations_logged:
                    violations_logged.add(log_key)
                    violations_log.append({
                        "frame":     frame_count,
                        "timestamp": f"{frame_count / video_fps:.1f}s",
                        "track_id":  track_id,
                        "type":      violation_type,
                        "speed":     round(avg_speed, 1),
                        "class":     track_class[track_id]
                    })

                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    ts   = datetime.now().strftime("%H%M%S_%f")[:10]
                    safe = violation_type.replace(" ", "_")
                    cv2.imwrite(
                        f"data/output/violations/{safe}_{track_id}_{ts}.jpg",
                        crop
                    )
                    print(f"  ⚠ [{frame_count / video_fps:.1f}s] {violation_type} CONFIRMED — "
                          f"{track_class[track_id]} #{track_id} @ {avg_speed:.1f} km/h")
            else:
                active_violations.pop(track_id, None)

            # 7. Drawing
            current_violation = active_violations.get(track_id)
            if current_violation:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(annotated_frame, "!!! VIOLATION !!!", (x1, max(y1 - 35, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                cv2.putText(annotated_frame, current_violation, (x1, max(y1 - 12, 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{track_class[track_id]} #{track_id}  {avg_speed:.1f} km/h"
            cv2.putText(annotated_frame, label, (x1, max(y1 - 55, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            draw_trail(annotated_frame, track_history, track_id)

    # HUD
    if violations_log:
        cv2.putText(annotated_frame, f"VIOLATIONS: {len(violations_log)}",
                    (20, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    out.write(annotated_frame)
    cv2.imshow("Violation Detector", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ Violation Detection Completed!")
print(f"Total Verified Violations: {len(violations_log)}")
if violations_log:
    print("\nAll Violations:")
    for v in violations_log:
        print(f"  • {v['timestamp']:>8} | {v['type']:>15} | "
              f"{v['class']} #{v['track_id']} @ {v['speed']} km/h")
print(f"\nViolation crops saved in: data/output/violations/")