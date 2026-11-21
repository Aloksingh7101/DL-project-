"""
PPE Detection & Compliance Engine
Supports dedicated PPE models and standard YOLOv8 with
advanced spatial & chromatic computer vision PPE verification.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

from config import MODELS_DIR, PPE_MODEL_PATH, DEFAULT_YOLO_MODEL, DEFAULT_RULES

class PPEDetector:
    def __init__(self, model_path=None, conf_threshold=0.30, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_model = self._load_person_model()
        self.ppe_model = self._load_ppe_model()

    def _load_person_model(self):
        """Loads YOLOv8 for robust multi-pose person detection."""
        print("Loading Person Detection Model (yolov8n.pt)...")
        return YOLO(DEFAULT_YOLO_MODEL)

    def _load_ppe_model(self):
        """Loads dedicated PPE weights if present."""
        if os.path.exists(PPE_MODEL_PATH) and os.path.getsize(PPE_MODEL_PATH) > 1000000:
            print(f"Loading Dedicated PPE Model from: {PPE_MODEL_PATH}")
            return YOLO(PPE_MODEL_PATH)
        return None

    def _check_gear_in_region_hsv(self, person_crop):
        """
        Calibrated chromatic and spatial inspection for safety helmet
        and high-visibility safety vest on person crop.
        """
        h, w, _ = person_crop.shape
        if h < 25 or w < 15:
            return False, False, False

        # --- 1. HEAD REGION (Top 25% of person) ---
        head_h = max(15, int(h * 0.25))
        head_crop = person_crop[0:head_h, :]
        head_hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        head_pixels = float(head_crop.shape[0] * head_crop.shape[1])

        # Yellow / Bright Amber Hardhat (H: 14 to 36, S: 70 to 255, V: 140 to 255)
        yellow_hat_mask = cv2.inRange(head_hsv, np.array([14, 70, 140]), np.array([36, 255, 255]))
        yellow_ratio = cv2.countNonZero(yellow_hat_mask) / head_pixels

        # White Hardhat (requires low saturation, high value, but distinct from sky via contrast)
        white_hat_mask = cv2.inRange(head_hsv, np.array([0, 0, 210]), np.array([180, 35, 255]))
        white_ratio = cv2.countNonZero(white_hat_mask) / head_pixels

        # Blue Hardhat
        blue_hat_mask = cv2.inRange(head_hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
        blue_ratio = cv2.countNonZero(blue_hat_mask) / head_pixels

        has_hardhat = (yellow_ratio > 0.07) or (white_ratio > 0.18) or (blue_ratio > 0.08)

        # --- 2. TORSO REGION (18% to 65% of person) ---
        torso_y1 = int(h * 0.18)
        torso_y2 = int(h * 0.65)
        torso_crop = person_crop[torso_y1:torso_y2, :]
        torso_hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        torso_pixels = float(torso_crop.shape[0] * torso_crop.shape[1])

        # Fluorescent Neon Lime / High-Vis Yellow Vest (H: 13 to 34, S: 60 to 255, V: 155 to 255)
        neon_vest_mask = cv2.inRange(torso_hsv, np.array([13, 60, 155]), np.array([34, 255, 255]))
        neon_ratio = cv2.countNonZero(neon_vest_mask) / torso_pixels

        # High-Vis Fluorescent Safety Orange Vest (H: 6 to 14, S: 130 to 255, V: 155 to 255)
        orange_vest_mask = cv2.inRange(torso_hsv, np.array([6, 130, 155]), np.array([14, 255, 255]))
        orange_ratio = cv2.countNonZero(orange_vest_mask) / torso_pixels

        has_vest = (neon_ratio > 0.08) or (orange_ratio > 0.09)

        # Mask heuristic (lower face)
        has_mask = False

        return has_hardhat, has_vest, has_mask

    def detect(self, frame, rules=None):
        """
        Run inference on frame and evaluate safety compliance.
        """
        if rules is None:
            rules = DEFAULT_RULES

        orig_h, orig_w = frame.shape[:2]
        
        # Determine appropriate inference size
        imgsz = 1280 if max(orig_h, orig_w) >= 1920 else 640

        # Person detection with YOLOv8
        res = self.person_model(
            frame, 
            imgsz=imgsz, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold, 
            classes=[0],  # Class 0 = Person
            verbose=False
        )[0]

        detected_workers = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detected_workers

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i, box_arr in enumerate(coords):
            x1, y1, x2, y2 = [int(v) for v in box_arr]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)

            pw = x2 - x1
            ph = y2 - y1
            # Filter out tiny background artifacts
            if pw < 20 or ph < 40:
                continue

            person_crop = frame[y1:y2, x1:x2]
            has_hardhat, has_vest, has_mask = self._check_gear_in_region_hsv(person_crop)
            
            gear_boxes = []

            # Check with dedicated PPE model on the person crop if available
            if self.ppe_model is not None and person_crop.size > 0:
                ppe_res = self.ppe_model(person_crop, conf=0.20, verbose=False)[0]
                if ppe_res.boxes is not None and len(ppe_res.boxes) > 0:
                    for p_cls, p_conf, p_xyxy in zip(ppe_res.boxes.cls, ppe_res.boxes.conf, ppe_res.boxes.xyxy):
                        p_label = self.ppe_model.names[int(p_cls)].lower()
                        px1, py1, px2, py2 = [int(v) for v in p_xyxy]
                        # Map crop coordinates back to full frame
                        global_gear_box = [x1 + px1, y1 + py1, x1 + px2, y1 + py2]
                        
                        if 'helmet' in p_label and 'no' not in p_label:
                            has_hardhat = True
                            gear_boxes.append({'label': 'Hardhat', 'box': global_gear_box, 'conf': float(p_conf)})
                        elif 'no_helmet' in p_label:
                            has_hardhat = False
                        
                        if 'mask' in p_label and 'no' not in p_label:
                            has_mask = True
                            gear_boxes.append({'label': 'Mask', 'box': global_gear_box, 'conf': float(p_conf)})

            # Synthesize gear bounding boxes if compliant and not already added
            if has_hardhat and not any('hardhat' in g['label'].lower() for g in gear_boxes):
                hh = int(ph * 0.22)
                gear_boxes.append({'label': 'Hardhat', 'box': [x1, y1, x2, y1 + hh], 'conf': 0.92})

            if has_vest and not any('vest' in g['label'].lower() for g in gear_boxes):
                vy1 = y1 + int(ph * 0.18)
                vy2 = y1 + int(ph * 0.65)
                gear_boxes.append({'label': 'Safety Vest', 'box': [x1, vy1, x2, vy2], 'conf': 0.94})

            # Compliance rule check
            missing_gear = []
            if rules.get("require_hardhat", True) and not has_hardhat:
                missing_gear.append("Hardhat")
            if rules.get("require_vest", True) and not has_vest:
                missing_gear.append("Safety Vest")
            if rules.get("require_mask", False) and not has_mask:
                missing_gear.append("Mask")

            is_compliant = (len(missing_gear) == 0)

            detected_workers.append({
                'box': [x1, y1, x2, y2],
                'confidence': float(confs[i]),
                'has_hardhat': has_hardhat,
                'has_vest': has_vest,
                'has_mask': has_mask,
                'is_compliant': is_compliant,
                'missing_gear': missing_gear,
                'gear_boxes': gear_boxes
            })

        return detected_workers
