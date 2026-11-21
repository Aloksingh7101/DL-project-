"""
Visualizer module for rendering real-time PPE detection and compliance overlays.
"""
import cv2
import numpy as np
from config import COLORS_BGR

def draw_hud_banner(frame, total_workers, compliant_count, violation_count):
    """Draw top HUD status banner with real-time compliance metrics."""
    h, w, _ = frame.shape
    banner_h = max(45, int(h * 0.05))
    overlay = frame.copy()
    
    # Semi-transparent dark banner
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLORS_BGR["dark_bg"], -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Compliance rate calculation
    rate = (compliant_count / total_workers * 100.0) if total_workers > 0 else 100.0
    status_color = COLORS_BGR["compliant"] if violation_count == 0 else COLORS_BGR["violation"]
    status_text = "ALL SAFE" if violation_count == 0 else f"{violation_count} VIOLATION{'S' if violation_count > 1 else ''} DETECTED"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, w / 2000.0)
    thick = max(1, int(w / 1200.0))

    # Left: Title and Status pill
    cv2.putText(frame, "PPE COMPLIANCE MONITOR", (20, int(banner_h * 0.65)), font, scale, (240, 240, 240), thick, cv2.LINE_AA)
    
    # Status badge
    badge_x = int(w * 0.35)
    badge_w = int(w * 0.18)
    badge_h = int(banner_h * 0.7)
    badge_y = int((banner_h - badge_h) / 2)
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), status_color, -1)
    cv2.putText(frame, status_text, (badge_x + 10, int(badge_y + badge_h * 0.7)), font, scale * 0.85, (255, 255, 255), thick, cv2.LINE_AA)

    # Right: KPI Stats
    stats_str = f"Workers: {total_workers} | Compliant: {compliant_count} | Violations: {violation_count} | Rate: {rate:.0f}%"
    text_size = cv2.getTextSize(stats_str, font, scale * 0.85, thick)[0]
    cv2.putText(frame, stats_str, (w - text_size[0] - 25, int(banner_h * 0.65)), font, scale * 0.85, (240, 240, 240), thick, cv2.LINE_AA)

    return frame

def draw_worker_annotations(frame, worker, show_gear_boxes=True):
    """
    Draw bounding boxes and status labels for a tracked worker.
    worker dict contains:
        'box', 'track_id', 'is_compliant', 'missing_gear', 'confidence', 'gear_boxes'
    """
    box = worker['box']
    is_compliant = worker['is_compliant']
    track_id = worker.get('track_id', 1)
    missing = worker.get('missing_gear', [])
    conf = worker.get('confidence', 0.0)
    gear_boxes = worker.get('gear_boxes', [])

    color = COLORS_BGR["compliant"] if is_compliant else COLORS_BGR["violation"]
    x1, y1, x2, y2 = [int(v) for v in box]

    # Calculate dynamic font scale and thickness based on frame resolution
    frame_w = frame.shape[1]
    thickness = max(2, int(frame_w / 600))
    font_scale = max(0.5, frame_w / 2200.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Main Worker Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 2. Status Badge Header
    if is_compliant:
        label = f"Worker #{track_id} | COMPLIANT ({int(conf * 100)}%)"
    else:
        missing_str = ", ".join(missing) if missing else "Safety Breach"
        label = f"Worker #{track_id} | VIOLATION: NO {missing_str.upper()}"

    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    badge_y1 = max(0, y1 - label_h - 10)
    badge_y2 = y1
    badge_x2 = min(frame.shape[1], x1 + label_w + 14)

    # Background box for label
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
    # Text
    cv2.putText(frame, label, (x1 + 7, y1 - 6), font, font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

    # 3. Gear specific boxes (if compliant and enabled)
    if show_gear_boxes and gear_boxes:
        for g in gear_boxes:
            gx1, gy1, gx2, gy2 = [int(v) for v in g['box']]
            g_lbl = g.get('label', 'Gear')
            g_color = COLORS_BGR["gear_helmet"] if 'hardhat' in g_lbl.lower() or 'helmet' in g_lbl.lower() else COLORS_BGR["gear_vest"]
            
            # Subtle inner rectangle
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), g_color, max(1, thickness - 1))
            g_tag = f"{g_lbl}"
            (gw, gh), _ = cv2.getTextSize(g_tag, font, font_scale * 0.7, 1)
            cv2.rectangle(frame, (gx1, gy1 - gh - 4), (gx1 + gw + 4, gy1), g_color, -1)
            cv2.putText(frame, g_tag, (gx1 + 2, gy1 - 2), font, font_scale * 0.7, (10, 10, 10), 1, cv2.LINE_AA)

    return frame
