# dashboard/dashboard.py
"""
TrafficVision AI — Unified Dashboard
Integrates: Detection & Speed Estimation · Traffic Density · Violation Detection
"""

import cv2
import csv
import os
import sys
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
from datetime import datetime

# Allow importing from sibling src/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    # Detection
    parse_box, update_class,
    # Speed
    estimate_speed, draw_speed,
    # Counting
    check_line_cross,
    # Density
    is_in_zone, get_density_level,
    # Drawing
    draw_trail,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH           = "yolov8m.pt"
VIDEO_PATH           = "data/input/traffic.mp4"
CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE             = 1280
TRACKER              = "bytetrack.yaml"
VIDEO_FPS            = 25.0

# Output paths
OUTPUT_DIR           = "data/output"
DASHBOARD_VIDEO      = os.path.join(OUTPUT_DIR, "dashboard_output.mp4")
VIOLATIONS_DIR       = os.path.join(OUTPUT_DIR, "violations")
CSV_LOG_PATH         = os.path.join(OUTPUT_DIR, "traffic_log.csv")

# ── Speed / Counting ──────────────────────────────────────────────────────────
LINE_Y_RATIO         = 0.65          # counting line at 65% of frame height

# ── Density ───────────────────────────────────────────────────────────────────
DENSITY_ZONE_RATIOS  = [             # polygon as (x%, y%) of frame
    (0.10, 0.25),
    (0.88, 0.25),
    (0.92, 0.72),
    (0.08, 0.72),
]

# ── Violations ────────────────────────────────────────────────────────────────
SPEED_LIMIT          = 60            # km/h
VIOLATION_FRAMES     = 20            # continuous frames required
SPEED_STD_MAX        = 10.0          # km/h — max std dev in violation window
SPEED_MARGIN         = 3.0           # km/h — must exceed limit by this margin
MIN_TRACK_AGE        = 30            # frames before violations can fire
WRONG_WAY_THRESHOLD  = -4.0          # negative dy = moving away from camera
WRONG_WAY_CONFIRM_N  = 8
WRONG_WAY_MIN_RATIO  = 0.75
SPEED_WINDOW_SIZE    = 20

# ── Dashboard Layout ──────────────────────────────────────────────────────────
PANEL_W              = 420           # right-side panel width in pixels

# ── Design System ─────────────────────────────────────────────────────────────
# Background layers
BG_PANEL        = (18,  18,  22)     # deep navy-black
BG_CARD         = (28,  28,  36)     # slightly lighter card
BG_CARD_HOVER   = (36,  36,  48)     # active/highlight card
BG_SEPARATOR    = (45,  45,  58)

# Accent palette (BGR)
ACCENT_AMBER    = (0,   180, 255)    # primary brand amber — BGR: orange
ACCENT_BLUE     = (220, 160,  60)    # cool blue for data
ACCENT_TEAL     = (180, 210,  80)    # teal for secondary info

# Status colours (BGR)
CLR_SUCCESS     = (100, 210,  80)    # green
CLR_WARNING     = (30,  190, 240)    # yellow-orange
CLR_DANGER      = (60,   60, 220)    # red
CLR_INFO        = (220, 160,  60)    # blue

# Text hierarchy (BGR)
TXT_PRIMARY     = (235, 235, 240)    # near-white
TXT_SECONDARY   = (160, 160, 175)    # muted
TXT_MUTED       = (95,   95, 115)    # very muted
TXT_HEADING     = (0,   180, 255)    # amber headings

# Video overlay colours (BGR)
CLR_BOX_NORMAL  = (100, 210,  80)    # green boxes
CLR_BOX_VIOL    = (60,   60, 220)    # red violation boxes
CLR_LINE        = (255, 210,  50)    # counting line
CLR_ZONE        = (200, 200,  80)    # density zone
CLR_TRAIL       = (200, 140,  60)    # track trail (blue-ish)
CLR_SPEED_OK    = (100, 210,  80)
CLR_SPEED_OVER  = (60,   60, 220)

# Legacy aliases (kept for compatibility with unchanged logic)
CLR_WHITE   = TXT_PRIMARY
CLR_GREEN   = CLR_SUCCESS
CLR_YELLOW  = CLR_WARNING
CLR_ORANGE  = (0,   165, 255)
CLR_RED     = CLR_DANGER
CLR_CYAN    = CLR_LINE
CLR_ACCENT  = ACCENT_AMBER
CLR_DIM     = TXT_MUTED


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION HELPERS  (exact logic from violation_detector.py)
# ─────────────────────────────────────────────────────────────────────────────

def is_overspeeding_verified(track_id, track_speed_counter, track_speed_window):
    if track_speed_counter[track_id] < VIOLATION_FRAMES:
        return False
    window = track_speed_window[track_id]
    if len(window) < VIOLATION_FRAMES:
        return False
    recent     = window[-VIOLATION_FRAMES:]
    mean_speed = np.mean(recent)
    std_speed  = np.std(recent)
    if mean_speed < SPEED_LIMIT + SPEED_MARGIN:
        return False
    if std_speed > SPEED_STD_MAX:
        return False
    return True


def is_wrong_way_verified(track_id, history, track_wrongway_counter):
    if track_wrongway_counter[track_id] < VIOLATION_FRAMES:
        return False
    if len(history) < WRONG_WAY_CONFIRM_N + 1:
        return False
    recent = history[-(WRONG_WAY_CONFIRM_N + 1):]
    wrong_way_count = sum(
        1 for i in range(1, len(recent))
        if recent[i][1] - recent[i - 1][1] < WRONG_WAY_THRESHOLD
    )
    return (wrong_way_count / WRONG_WAY_CONFIRM_N) >= WRONG_WAY_MIN_RATIO


# ─────────────────────────────────────────────────────────────────────────────
# UI PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _text(frame, msg, x, y, scale=0.55, color=TXT_PRIMARY, thickness=1, bold=False):
    """Render anti-aliased text. bold=True uses thickness+1."""
    t = thickness + 1 if bold else thickness
    cv2.putText(frame, str(msg), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)


def _text_right(frame, msg, right_x, y, scale=0.5, color=TXT_PRIMARY, thickness=1):
    """Right-aligned text ending at right_x."""
    (tw, _), _ = cv2.getTextSize(str(msg), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    _text(frame, msg, right_x - tw, y, scale, color, thickness)


def _card(frame, x, y, w, h, color=BG_CARD, radius=6):
    """Filled rounded-rect card via overlapping rects (OpenCV 4.x compatible)."""
    # Approximate rounded rect with a filled rect + smaller corner rects masked
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in [(x + radius, y + radius),
                   (x + w - radius, y + radius),
                   (x + radius, y + h - radius),
                   (x + w - radius, y + h - radius)]:
        cv2.circle(frame, (cx, cy), radius, color, -1)


def _card_outline(frame, x, y, w, h, color, radius=6, thickness=1):
    """Outline rounded-rect — drawn as 4 lines + 4 arcs."""
    cv2.line(frame, (x + radius, y),     (x + w - radius, y),     color, thickness)
    cv2.line(frame, (x + radius, y + h), (x + w - radius, y + h), color, thickness)
    cv2.line(frame, (x, y + radius),     (x, y + h - radius),     color, thickness)
    cv2.line(frame, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
    for (cx, cy, start, end) in [
        (x + radius,     y + radius,     180, 270),
        (x + w - radius, y + radius,     270, 360),
        (x + radius,     y + h - radius, 90,  180),
        (x + w - radius, y + h - radius, 0,   90),
    ]:
        cv2.ellipse(frame, (cx, cy), (radius, radius),
                    0, start, end, color, thickness)


def _hbar(frame, x, y, w, h, fill_ratio, fg_color, bg_color=BG_SEPARATOR, radius=3):
    """Rounded progress bar."""
    # Background track
    _card(frame, x, y, w, h, bg_color, radius)
    fill = int(w * min(max(fill_ratio, 0.0), 1.0))
    if fill > radius * 2:
        _card(frame, x, y, fill, h, fg_color, radius)
    elif fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), fg_color, -1)


def _divider(frame, x, y, w, color=BG_SEPARATOR):
    cv2.line(frame, (x, y), (x + w, y), color, 1)


def _badge(frame, x, y, w, h, text, bg_color, text_color=TXT_PRIMARY, radius=5):
    """Filled badge with centered text."""
    _card(frame, x, y, w, h, bg_color, radius)
    scale = 0.58
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2, cv2.LINE_AA)


def _dot(frame, cx, cy, radius, color):
    cv2.circle(frame, (cx, cy), radius, color, -1)


def _section_label(frame, text, x, y, panel_w):
    """Styled section label: small caps with left accent bar."""
    # Left accent bar
    cv2.rectangle(frame, (x, y - 10), (x + 3, y + 4), ACCENT_AMBER, -1)
    _text(frame, text, x + 10, y, scale=0.48, color=TXT_HEADING, thickness=2)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_header(panel, x, y, pw, frame_count, video_fps):
    """Logo bar + timestamp."""
    # Top gradient bar (simulated with rectangle strips)
    for i in range(4):
        alpha = 1.0 - i * 0.22
        c = int(255 * alpha)
        cv2.rectangle(panel, (x, y + i), (x + pw, y + i + 1),
                      (0, int(c * 0.70), int(c * 1.0)), -1)
    y += 6

    # Logo text
    _text(panel, "TRAFFIC", x + 14, y + 22,
          scale=0.8, color=TXT_PRIMARY, thickness=2)
    _text(panel, "VISION", x + 14, y + 40,
          scale=0.8, color=ACCENT_AMBER, thickness=2)
    _text(panel, "AI", x + 14, y + 58,
          scale=0.8, color=TXT_PRIMARY, thickness=2)

    # Right-side: system status dot + time
    ts = datetime.now().strftime("%H:%M:%S")
    _dot(panel, x + pw - 16, y + 16, 5, CLR_SUCCESS)
    _text(panel, "LIVE", x + pw - 48, y + 20,
          scale=0.38, color=CLR_SUCCESS)

    _text(panel, ts, x + pw - 72, y + 36,
          scale=0.45, color=TXT_PRIMARY, bold=True)

    elapsed_s = frame_count / max(video_fps, 1)
    elapsed_str = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"
    _text(panel, f"ELAPSED  {elapsed_str}", x + pw - 92, y + 52,
          scale=0.38, color=TXT_MUTED)

    # Version tag bottom-right
    _text(panel, "PHASE 1", x + pw - 52, y + 68,
          scale=0.33, color=TXT_MUTED)

    return y + 82   # next y


def _draw_counts(panel, x, y, pw, vehicle_counter, total_counted):
    """Vehicle count section with per-class rows.

    Layout (left to right, fixed columns so nothing can collide):
      [ label 56px ] [ count 42px, right-aligned ] [ gap 10px ] [ bar -> fills rest ]
    """
    SECTION_H = 154
    _card(panel, x + 8, y, pw - 16, SECTION_H, BG_CARD, radius=8)
    _section_label(panel, "VEHICLE COUNTS", x + 16, y + 16, pw)

    classes = [
        ("car",        "Car",        CLR_INFO),
        ("motorcycle", "Moto",       ACCENT_TEAL),
        ("bus",        "Bus",        CLR_WARNING),
        ("truck",      "Truck",      ACCENT_AMBER),
    ]

    label_x      = x + 18
    count_right  = x + 78     # count text right-edge (fixed column)
    bar_x        = x + 92     # bar starts well clear of the count column
    bar_w        = (x + pw - 18) - bar_x
    bar_h        = 10

    row_y = y + 34
    for cls, label, color in classes:
        count = vehicle_counter.get(cls, 0)
        ratio = count / max(total_counted, 1)

        # 1) bar drawn first (background layer)
        _hbar(panel, bar_x, row_y - 9, bar_w, bar_h, ratio, color)

        # 2) label and count drawn after, on top — never overlapped by the bar
        _text(panel, label, label_x, row_y,
              scale=0.42, color=TXT_SECONDARY)
        _text_right(panel, str(count), count_right, row_y,
                    scale=0.46, color=TXT_PRIMARY, thickness=2)

        row_y += 26

    # Total
    _divider(panel, x + 18, row_y, pw - 36)
    row_y += 22
    _text(panel, "TOTAL", x + 18, row_y,
          scale=0.44, color=TXT_MUTED)
    _text_right(panel, str(total_counted), x + pw - 18, row_y + 2,
                scale=0.62, color=TXT_PRIMARY, thickness=2)

    return y + SECTION_H + 10


def _draw_speed(panel, x, y, pw, avg_speed):
    """Speed monitor section."""
    SECTION_H = 100
    _card(panel, x + 8, y, pw - 16, SECTION_H, BG_CARD, radius=8)
    _section_label(panel, "SPEED MONITOR", x + 16, y + 16, pw)

    over_limit = avg_speed > SPEED_LIMIT
    spd_color  = CLR_DANGER if over_limit else CLR_SUCCESS
    status_txt = "OVER LIMIT" if over_limit else "NORMAL"
    status_clr = CLR_DANGER if over_limit else CLR_SUCCESS

    # Large speed readout
    spd_str = f"{avg_speed:05.1f}"
    _text(panel, spd_str, x + 18, y + 55,
          scale=1.0, color=spd_color, bold=True)
    _text(panel, "km/h  avg", x + 110, y + 56,
          scale=0.44, color=TXT_MUTED)

    # Status pill
    pill_x = x + pw - 105
    _badge(panel, pill_x, y + 38, 88, 22, status_txt,
           CLR_DANGER if over_limit else (20, 60, 25), status_clr)

    # Limit label + bar
    _text(panel, f"Limit  {SPEED_LIMIT} km/h", x + 18, y + 72,
          scale=0.40, color=TXT_MUTED)
    _hbar(panel, x + 18, y + 78, pw - 40, 8,
          avg_speed / 130.0, spd_color)

    return y + SECTION_H + 10


def _draw_density(panel, x, y, pw, active_in_zone,
                  density_level, density_color, density_desc):
    """Traffic density section."""
    SECTION_H = 124
    _card(panel, x + 8, y, pw - 16, SECTION_H, BG_CARD, radius=8)
    _section_label(panel, "TRAFFIC DENSITY", x + 16, y + 16, pw)

    # Density badge (big pill)
    badge_w = pw - 36
    badge_h = 30
    bx = x + 18
    by = y + 24
    _card(panel, bx, by, badge_w, badge_h, density_color, radius=6)
    # Subtle overlay for depth
    overlay = np.zeros((badge_h, badge_w, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (0, 0), (badge_w, badge_h // 2),
                  (255, 255, 255), -1)
    # Blend at ~8% opacity
    panel_roi = panel[by:by + badge_h, bx:bx + badge_w]
    cv2.addWeighted(panel_roi, 1.0, overlay, 0.07, 0, panel_roi)

    # Badge text centered
    scale = 0.65
    (tw, _), _ = cv2.getTextSize(density_level,
                                  cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(panel, density_level,
                (bx + (badge_w - tw) // 2, by + 21),
                cv2.FONT_HERSHEY_SIMPLEX, scale, TXT_PRIMARY, 2, cv2.LINE_AA)

    # Zone count (left, own line) and description (right, own line) —
    # stacked vertically so long strings never collide, no char-width guessing
    _text(panel, f"{active_in_zone} vehicles in zone",
          x + 18, y + 72, scale=0.42, color=TXT_SECONDARY)
    _text_right(panel, density_desc, x + pw - 18, y + 88,
                scale=0.40, color=TXT_MUTED)

    # Fill bar
    _hbar(panel, x + 18, y + 96, pw - 36, 8,
          active_in_zone / 15.0, density_color)

    return y + SECTION_H + 10


def _draw_violations(panel, x, y, pw, ph, violations_log):
    """Violations section — fills remaining panel height."""
    avail_h = ph - y - 36   # leave room for footer
    avail_h = max(avail_h, 60)
    _card(panel, x + 8, y, pw - 16, avail_h, BG_CARD, radius=8)
    _section_label(panel, "VIOLATIONS", x + 16, y + 16, pw)

    total_v = len(violations_log)
    v_color = CLR_DANGER if total_v > 0 else CLR_SUCCESS

    # Total count badge (top-right of section)
    badge_txt = str(total_v)
    _badge(panel, x + pw - 50, y + 6, 36, 20,
           badge_txt, CLR_DANGER if total_v > 0 else (18, 55, 20),
           TXT_PRIMARY)

    if total_v == 0:
        _text(panel, "No violations detected",
              x + 18, y + 50, scale=0.44, color=TXT_MUTED)
        return

    # How many rows fit?
    ROW_H    = 46
    max_rows = max(1, (avail_h - 28) // ROW_H)
    recent_v = violations_log[-max_rows:]

    ry = y + 26
    for v in reversed(recent_v):
        if ry + ROW_H > y + avail_h - 4:
            break

        vtype = v["type"]
        spd   = v["speed"]
        tid   = v["track_id"]
        ts_v  = v["timestamp"]
        vcls  = v.get("class", "?")

        # Row card
        _card(panel, x + 14, ry, pw - 28, ROW_H - 4,
              (38, 22, 22), radius=5)
        _card_outline(panel, x + 14, ry, pw - 28, ROW_H - 4,
                      (80, 30, 30), radius=5, thickness=1)

        # Left accent bar for violation type
        bar_clr = CLR_DANGER
        cv2.rectangle(panel, (x + 14, ry + 4),
                      (x + 17, ry + ROW_H - 8), bar_clr, -1)

        # Type + ID
        _text(panel, vtype, x + 24, ry + 14,
              scale=0.46, color=CLR_DANGER, bold=True)
        _text(panel, f"#{tid} · {vcls}", x + 24, ry + 28,
              scale=0.38, color=TXT_MUTED)

        # Right side: speed + time
        _text_right(panel, f"{spd} km/h",
                    x + pw - 18, ry + 14,
                    scale=0.44, color=TXT_PRIMARY, thickness=1)
        _text_right(panel, ts_v,
                    x + pw - 18, ry + 28,
                    scale=0.37, color=TXT_MUTED)

        ry += ROW_H


def _draw_footer(panel, x, y, pw):
    _divider(panel, x + 14, y - 10, pw - 28, BG_SEPARATOR)
    _text(panel, "TrafficVision AI  |  Phase 1",
          x + 18, y + 6, scale=0.35, color=TXT_MUTED)
    _text_right(panel, "YOLOv8m  ·  ByteTrack",
                x + pw - 14, y + 6, scale=0.33, color=TXT_MUTED)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(panel_frame, panel_x,
                frame_count, video_fps,
                vehicle_counter, total_counted,
                active_in_zone, density_level, density_color, density_desc,
                avg_speed_all, violations_log, active_violations,
                frame_height):
    """
    Renders the full right-side info panel onto panel_frame.
    panel_x is the left edge of the panel in the combined canvas.
    """
    pw = PANEL_W
    ph = frame_height
    x  = panel_x

    # ── Panel base ────────────────────────────────────────────────────────────
    cv2.rectangle(panel_frame, (x, 0), (x + pw, ph), BG_PANEL, -1)

    # Subtle vertical separator with gradient feel
    for i in range(3):
        alpha = 1.0 - i * 0.4
        clr   = int(80 * alpha)
        cv2.line(panel_frame, (x + i, 0), (x + i, ph), (clr, clr, clr), 1)

    # ── Sections ─────────────────────────────────────────────────────────────
    y = 8
    y = _draw_header(panel_frame, x, y, pw, frame_count, video_fps)
    y += 6
    _divider(panel_frame, x + 14, y, pw - 28, BG_SEPARATOR)
    y += 10

    y = _draw_counts(panel_frame, x, y, pw, vehicle_counter, total_counted)
    y = _draw_speed(panel_frame, x, y, pw, avg_speed_all)
    y = _draw_density(panel_frame, x, y, pw,
                      active_in_zone, density_level, density_color, density_desc)

    _draw_violations(panel_frame, x, y, pw, ph, violations_log)
    _draw_footer(panel_frame, x, ph - 20, pw)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_on_video(frame, x1, y1, x2, y2,
                  track_id, track_class, avg_speed,
                  violation_type, track_history):
    """Draw box, label, speed, trail on the video portion of the canvas."""

    if violation_type:
        box_color = CLR_BOX_VIOL
        thickness = 3

        # Semi-transparent red fill for violation box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 60), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # Violation banner above box
        banner_h = 36
        banner_y = max(y1 - banner_h - 2, 0)
        banner_x1 = max(x1, 0)
        banner_x2 = min(x2, frame.shape[1])
        _card(frame, banner_x1, banner_y, banner_x2 - banner_x1, banner_h,
              (0, 0, 160), radius=4)
        _text(frame, "VIOLATION", banner_x1 + 6, banner_y + 13,
              scale=0.44, color=TXT_PRIMARY, bold=True)
        _text(frame, violation_type, banner_x1 + 6, banner_y + 28,
              scale=0.40, color=CLR_DANGER, bold=True)

    else:
        box_color = CLR_BOX_NORMAL
        thickness = 2

    # Corner-style bounding box (professional look)
    corner = 12
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner, y1), box_color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), box_color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner, y1), box_color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), box_color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner, y2), box_color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), box_color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner, y2), box_color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), box_color, thickness)

    # ── Label: two distinct chips side by side — class/ID, then speed ────────
    id_label = f"{track_class} #{track_id}" if track_class and track_class != "?" \
               else f"#{track_id}"

    pad_x   = 6
    gap     = 4   # gap between the two chips so they never read as one string

    (id_w, id_h), _   = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    chip_h  = id_h + 10
    chip_y  = max(y1 - chip_h - 4, 0)
    chip_x  = max(x1, 0)

    # ID / class chip
    id_chip_w = id_w + pad_x * 2
    cv2.rectangle(frame, (chip_x, chip_y),
                  (chip_x + id_chip_w, chip_y + chip_h), (15, 15, 15), -1)
    _text(frame, id_label, chip_x + pad_x, chip_y + chip_h - 7,
          scale=0.44, color=box_color)

    # Speed chip — separate rectangle, separate background, clear gap
    if avg_speed is not None:
        spd_clr  = CLR_SPEED_OVER if avg_speed > SPEED_LIMIT else box_color
        spd_text = f"{avg_speed:.1f} km/h"
        (spd_w, _), _ = cv2.getTextSize(spd_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        spd_chip_x = chip_x + id_chip_w + gap
        spd_chip_w = spd_w + pad_x * 2
        spd_chip_end = min(spd_chip_x + spd_chip_w, frame.shape[1])

        cv2.rectangle(frame, (spd_chip_x, chip_y),
                      (spd_chip_end, chip_y + chip_h), (15, 15, 15), -1)
        _text(frame, spd_text, spd_chip_x + pad_x, chip_y + chip_h - 7,
              scale=0.44, color=spd_clr)

    # Trail
    if len(track_history[track_id]) > 1:
        points = track_history[track_id]
        for i in range(1, len(points)):
            alpha   = i / len(points)
            t_color = tuple(int(c * alpha) for c in CLR_TRAIL)
            cv2.line(frame, points[i - 1], points[i], t_color, 1)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO FRAME OVERLAYS
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame_overlays(frame, LINE_Y, DENSITY_ZONE, frame_width,
                        frame_count, video_fps):
    """Draw counting line, density zone, and frame info onto video."""

    # ── Counting line ─────────────────────────────────────────────────────────
    # Dashed line effect
    dash_on = True
    for sx in range(60, frame_width - 60, 18):
        if dash_on:
            cv2.line(frame, (sx, LINE_Y), (sx + 10, LINE_Y), CLR_LINE, 2)
        dash_on = not dash_on
    # End caps
    cv2.line(frame, (50, LINE_Y - 6), (50, LINE_Y + 6), CLR_LINE, 2)
    cv2.line(frame, (frame_width - 50, LINE_Y - 6),
             (frame_width - 50, LINE_Y + 6), CLR_LINE, 2)
    # Label
    _card(frame, 54, LINE_Y - 18, 82, 16, (10, 10, 10), radius=3)
    _text(frame, "COUNT LINE", 58, LINE_Y - 5,
          scale=0.35, color=CLR_LINE)

    # ── Density zone ──────────────────────────────────────────────────────────
    # Subtle fill
    zone_overlay = frame.copy()
    cv2.fillPoly(zone_overlay, [DENSITY_ZONE], (20, 20, 60))
    cv2.addWeighted(zone_overlay, 0.12, frame, 0.88, 0, frame)
    # Dashed outline
    pts = DENSITY_ZONE.tolist()
    for i in range(len(pts)):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % len(pts)])
        cv2.line(frame, p1, p2, CLR_ZONE, 1)
    # Corner dots
    for pt in pts:
        _dot(frame, pt[0], pt[1], 4, CLR_ZONE)

    # ── Top-left frame info ───────────────────────────────────────────────────
    info_bg_h = 28
    cv2.rectangle(frame, (0, 0), (220, info_bg_h), (10, 10, 10), -1)
    elapsed   = frame_count / max(video_fps, 1)
    _text(frame, f"FRAME {frame_count:>5}",
          8, 17, scale=0.42, color=TXT_MUTED)
    _text(frame, f"{elapsed:.1f}s",
          100, 17, scale=0.42, color=TXT_SECONDARY)
    _dot(frame, 155, 12, 4, CLR_SUCCESS)
    _text(frame, "ANALYZING", 164, 17, scale=0.38, color=CLR_SUCCESS)


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow([
        "timestamp_s", "frame", "event_type",
        "track_id", "vehicle_class",
        "speed_kmh", "violation_type",
        "density_level", "vehicles_in_zone"
    ])
    return f, writer


def log_event(writer, frame_count, video_fps, event_type,
              track_id="", vehicle_class="", speed_kmh="",
              violation_type="", density_level="", vehicles_in_zone=""):
    writer.writerow([
        f"{frame_count / video_fps:.2f}",
        frame_count,
        event_type,
        track_id,
        vehicle_class,
        speed_kmh,
        violation_type,
        density_level,
        vehicles_in_zone,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)

    model     = YOLO(MODEL_PATH)
    cap       = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS

    # Canvas = video frame + right panel
    canvas_w = frame_width + PANEL_W
    canvas_h = frame_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(
        DASHBOARD_VIDEO, fourcc, video_fps, (canvas_w, canvas_h)
    )

    csv_file, csv_writer = init_csv(CSV_LOG_PATH)

    # ── Derived geometry ─────────────────────────────────────────────────────
    LINE_Y = int(frame_height * LINE_Y_RATIO)

    DENSITY_ZONE = np.array([
        [int(frame_width * rx), int(frame_height * ry)]
        for rx, ry in DENSITY_ZONE_RATIOS
    ], np.int32)

    # ── Shared tracking state ────────────────────────────────────────────────
    track_history     = defaultdict(list)
    track_class_votes = defaultdict(list)
    track_class       = {}
    track_start_frame = {}
    track_speeds      = defaultdict(list)

    # Speed / counting (from speed_estimator.py)
    vehicle_counter   = Counter()
    track_crossed     = set()

    # Violation (from violation_detector.py)
    track_speed_counter    = defaultdict(int)
    track_wrongway_counter = defaultdict(int)
    track_speed_window     = defaultdict(list)
    violations_logged      = set()
    violations_log         = []
    active_violations      = {}

    # Runtime speed accumulator for panel avg
    all_speeds_this_frame  = []

    frame_count   = 0
    panel_avg_spd = 0.0
    last_density  = ("LOW", CLR_SUCCESS, "Light Traffic")
    last_zone_cnt = 0

    print(f"\n🚦 TrafficVision AI Dashboard starting...")
    print(f"   Video  : {VIDEO_PATH}")
    print(f"   Output : {DASHBOARD_VIDEO}")
    print(f"   CSV    : {CSV_LOG_PATH}")
    print(f"   Violations dir: {VIOLATIONS_DIR}\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        all_speeds_this_frame.clear()

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.5,
            verbose=False
        )

        # video_frame is what we draw on (left side of canvas)
        video_frame = frame.copy()

        # Draw static overlays (line, zone, frame info)
        draw_frame_overlays(video_frame, LINE_Y, DENSITY_ZONE,
                            frame_width, frame_count, video_fps)

        active_in_zone = 0

        if results[0].boxes.id is not None:
            for box in results[0].boxes:

                # ── 1. Detection (speed_estimator.py logic) ─────────────────
                b = parse_box(box)
                if b is None:
                    continue

                track_id = b["track_id"]
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

                track_history[track_id].append(b["center"])
                if len(track_history[track_id]) > 25:
                    track_history[track_id].pop(0)

                if track_id not in track_start_frame:
                    track_start_frame[track_id] = frame_count

                # ── 2. Class voting (speed_estimator.py logic) ───────────────
                update_class(track_id, b["cls_id"], b["confidence"],
                             track_class_votes, track_class)

                # ── 3. Density zone count (density_estimator.py logic) ───────
                age = frame_count - track_start_frame[track_id]
                if age >= 8 and is_in_zone(b["center"], DENSITY_ZONE):
                    active_in_zone += 1

                # ── 4. Skip immature tracks for speed/violations ─────────────
                if age < 10:
                    draw_on_video(video_frame, x1, y1, x2, y2,
                                  track_id, track_class.get(track_id, "?"),
                                  None, None, track_history)
                    continue

                # ── 5. Speed estimation (speed_estimator.py logic) ───────────
                avg_speed = estimate_speed(
                    track_id, track_history, track_speeds,
                    video_fps, frame_height
                )

                if avg_speed is not None:
                    all_speeds_this_frame.append(avg_speed)

                # ── 6. Counting (speed_estimator.py logic) ───────────────────
                crossed_now = track_id not in track_crossed
                check_line_cross(track_id, track_history, track_class,
                                 track_crossed, vehicle_counter, LINE_Y)
                if crossed_now and track_id in track_crossed:
                    log_event(csv_writer, frame_count, video_fps,
                              "LINE_CROSS",
                              track_id=track_id,
                              vehicle_class=track_class.get(track_id, ""),
                              speed_kmh=f"{avg_speed:.1f}" if avg_speed else "")

                # ── 7. Violation detection (violation_detector.py logic) ──────
                violation_type = None

                if age >= MIN_TRACK_AGE and avg_speed is not None:
                    # Speed window for stability
                    track_speed_window[track_id].append(avg_speed)
                    if len(track_speed_window[track_id]) > SPEED_WINDOW_SIZE:
                        track_speed_window[track_id].pop(0)

                    # Overspeeding counter
                    if avg_speed > SPEED_LIMIT:
                        track_speed_counter[track_id] += 1
                    else:
                        track_speed_counter[track_id] = 0

                    # Wrong way counter
                    history = track_history[track_id]
                    if len(history) >= 6:
                        dy = history[-1][1] - history[-6][1]
                        if dy < WRONG_WAY_THRESHOLD:
                            track_wrongway_counter[track_id] += 1
                        else:
                            track_wrongway_counter[track_id] = 0

                    # Verify
                    if is_overspeeding_verified(track_id,
                                                track_speed_counter,
                                                track_speed_window):
                        violation_type = "OVERSPEEDING"

                    if is_wrong_way_verified(track_id,
                                             track_history[track_id],
                                             track_wrongway_counter):
                        violation_type = "WRONG WAY"

                # ── 8. Log & save violation once ─────────────────────────────
                if violation_type:
                    active_violations[track_id] = violation_type
                    log_key = (track_id, violation_type)

                    if log_key not in violations_logged:
                        violations_logged.add(log_key)
                        spd_val = round(avg_speed, 1) if avg_speed else 0.0
                        violations_log.append({
                            "frame":     frame_count,
                            "timestamp": f"{frame_count / video_fps:.1f}s",
                            "track_id":  track_id,
                            "type":      violation_type,
                            "speed":     spd_val,
                            "class":     track_class.get(track_id, "?"),
                        })

                        # Crop — add small padding around the vehicle
                        pad  = 20
                        cy1  = max(0, y1 - pad)
                        cy2  = min(frame_height, y2 + pad)
                        cx1  = max(0, x1 - pad)
                        cx2  = min(frame_width,  x2 + pad)
                        crop = frame[cy1:cy2, cx1:cx2]

                        ts_str = datetime.now().strftime("%H%M%S_%f")[:10]
                        safe   = violation_type.replace(" ", "_")
                        crop_path = os.path.join(
                            VIOLATIONS_DIR,
                            f"{safe}_{track_id}_{ts_str}.jpg"
                        )
                        cv2.imwrite(crop_path, crop)

                        log_event(csv_writer, frame_count, video_fps,
                                  "VIOLATION",
                                  track_id=track_id,
                                  vehicle_class=track_class.get(track_id, ""),
                                  speed_kmh=f"{spd_val:.1f}",
                                  violation_type=violation_type)

                        print(f"  ⚠  [{frame_count / video_fps:.1f}s] "
                              f"{violation_type} CONFIRMED — "
                              f"{track_class.get(track_id,'?')} "
                              f"#{track_id} @ {spd_val:.1f} km/h")
                else:
                    active_violations.pop(track_id, None)

                # ── 9. Draw on video frame ────────────────────────────────────
                draw_on_video(
                    video_frame, x1, y1, x2, y2,
                    track_id,
                    track_class.get(track_id, "?"),
                    avg_speed,
                    active_violations.get(track_id),
                    track_history
                )

        # ── Density (density_estimator.py logic) ─────────────────────────────
        last_density  = get_density_level(active_in_zone)
        last_zone_cnt = active_in_zone

        density_level, density_color, density_desc = last_density

        # Log density every 25 frames
        if frame_count % 25 == 0:
            log_event(csv_writer, frame_count, video_fps,
                      "DENSITY_SAMPLE",
                      density_level=density_level,
                      vehicles_in_zone=active_in_zone)

        # ── Panel avg speed ───────────────────────────────────────────────────
        if all_speeds_this_frame:
            panel_avg_spd = float(np.mean(all_speeds_this_frame))

        # ── Compose canvas ───────────────────────────────────────────────────
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :frame_width] = video_frame

        total_counted = sum(vehicle_counter.values())

        build_panel(
            canvas, frame_width,
            frame_count, video_fps,
            dict(vehicle_counter), total_counted,
            active_in_zone, density_level, density_color, density_desc,
            panel_avg_spd,
            violations_log, active_violations,
            frame_height
        )

        out_writer.write(canvas)
        cv2.imshow("TrafficVision AI Dashboard", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹  Stopped by user.")
            break

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    csv_file.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  ✅  TrafficVision AI — Run Complete")
    print("═" * 55)
    print(f"  Frames processed  : {frame_count}")
    print(f"  Vehicles counted  : {dict(vehicle_counter)}")
    print(f"  Violations found  : {len(violations_log)}")
    print(f"\n  Dashboard video   : {DASHBOARD_VIDEO}")
    print(f"  CSV log           : {CSV_LOG_PATH}")
    print(f"  Violation crops   : {VIOLATIONS_DIR}/")

    if violations_log:
        print("\n  Confirmed Violations:")
        for v in violations_log:
            print(f"    • {v['timestamp']:>8}  |  {v['type']:<15}  |  "
                  f"{v['class']} #{v['track_id']}  @  {v['speed']} km/h")

    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
