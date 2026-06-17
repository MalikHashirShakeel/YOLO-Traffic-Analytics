# src/utils.py

import cv2
import numpy as np
from collections import defaultdict, Counter

# ── Constants shared across all scripts ──────────────────────────────────────
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

MIN_BOX_WIDTH  = 25
MIN_BOX_HEIGHT = 25


# ─────────────────────────────────────────────────────────────────────────────
# 1. DETECTION — filter & parse one YOLO box
# ─────────────────────────────────────────────────────────────────────────────

def parse_box(box):
    """
    Parse a single YOLO box. Returns a dict or None if the box should be skipped.

    Returns:
        {
            "cls_id":     int,
            "confidence": float,
            "x1","y1","x2","y2": int,
            "width","height":    int,
            "center":     (int, int),
            "track_id":   int
        }
        or None if the box fails class / size filters.
    """
    cls_id = int(box.cls[0])
    if cls_id not in VEHICLE_CLASSES:
        return None

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    width  = x2 - x1
    height = y2 - y1

    if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
        return None

    return {
        "cls_id":     cls_id,
        "confidence": float(box.conf[0]),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width":  width,
        "height": height,
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        "track_id": int(box.id[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASS VOTING — confidence-weighted majority vote over a sliding window
# ─────────────────────────────────────────────────────────────────────────────

def update_class(track_id, cls_id, confidence,
                 track_class_votes, track_class,
                 window=15):
    """
    Append (cls_id, confidence) to the vote history for track_id,
    resolve the winning class, and store it in track_class[track_id].

    Args:
        track_id:          int
        cls_id:            int  — raw YOLO class from this frame
        confidence:        float
        track_class_votes: defaultdict(list)  — mutated in place
        track_class:       dict               — mutated in place
        window:            int  — how many recent frames to keep (default 15)
    """
    track_class_votes[track_id].append((cls_id, confidence))
    if len(track_class_votes[track_id]) > window:
        track_class_votes[track_id].pop(0)

    class_scores = defaultdict(float)
    for cid, conf in track_class_votes[track_id]:
        class_scores[cid] += conf

    best_cls = max(class_scores, key=class_scores.get)
    track_class[track_id] = VEHICLE_CLASSES[best_cls]


# ─────────────────────────────────────────────────────────────────────────────
# 3. SPEED ESTIMATION — pixels-per-meter + smoothed speed
# ─────────────────────────────────────────────────────────────────────────────

def get_pixels_per_meter(y_position, frame_height):
    """
    Perspective-corrected PPM for a forward-facing traffic camera.
    Calibrated so speed stays consistent (~55-60 km/h) across the full frame.
      - Top    (ratio~0.0): ~2.5  PPM
      - Middle (ratio~0.5): ~50   PPM
      - Bottom (ratio~1.0): ~365  PPM
    """
    ratio = y_position / frame_height
    BASE_PPM = 2.5
    MAX_PPM  = 365.0
    EXPONENT = 2.4
    return max(BASE_PPM, BASE_PPM + (MAX_PPM - BASE_PPM) * (ratio ** EXPONENT))


def estimate_speed(track_id, track_history, track_speeds, video_fps, frame_height):
    """
    Compute the smoothed speed (km/h) for track_id from its recent position history.
    Mirrors the exact logic in speed_estimator.py.

    Args:
        track_id:      int
        track_history: defaultdict(list) of (x, y) centres
        track_speeds:  defaultdict(list) of recent speed samples — mutated in place
        video_fps:     float
        frame_height:  int

    Returns:
        avg_speed (float) if enough history exists and time has elapsed,
        otherwise None.
    """
    history = track_history[track_id]
    if len(history) < 6:
        return None

    recent_points  = np.array(history[-6:])
    pixel_distance = np.linalg.norm(recent_points[-1] - recent_points[0])

    frames_elapsed = len(recent_points) - 1
    time_elapsed   = frames_elapsed / video_fps
    if time_elapsed <= 0:
        return None

    avg_y        = np.mean(recent_points[:, 1])
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

    return float(np.mean(track_speeds[track_id]))


# ─────────────────────────────────────────────────────────────────────────────
# 4. COUNTING — check if a track just crossed LINE_Y
# ─────────────────────────────────────────────────────────────────────────────

def check_line_cross(track_id, track_history, track_class,
                     track_crossed, vehicle_counter, line_y):
    """
    If track_id's centre just crossed line_y (upward → downward),
    increment vehicle_counter and mark the track as crossed.

    Args:
        track_id:        int
        track_history:   defaultdict(list)
        track_class:     dict
        track_crossed:   set  — mutated in place
        vehicle_counter: Counter — mutated in place
        line_y:          int
    """
    if track_id in track_crossed:
        return

    history_y = [p[1] for p in track_history[track_id]]
    if len(history_y) >= 2 and history_y[-2] < line_y <= history_y[-1]:
        vehicle_counter[track_class[track_id]] += 1
        track_crossed.add(track_id)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_box_label(frame, x1, y1, x2, y2, label,
                   box_color=(0, 255, 0), text_color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


def draw_trail(frame, track_history, track_id, color=(0, 165, 255)):
    if len(track_history[track_id]) > 1:
        points = np.array(track_history[track_id],
                          dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, color, 2)


def draw_speed(frame, avg_speed, x1, y2):
    speed_text = f"{avg_speed:.1f} km/h"
    color = (0, 255, 0) if avg_speed > 8 else (0, 180, 255)
    cv2.putText(frame, speed_text, (x1, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)