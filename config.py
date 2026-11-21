"""
Configuration and constants for Real-Time PPE Detection & Safety Compliance Monitoring
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SNAPSHOTS_DIR = os.path.join(OUTPUTS_DIR, "snapshots")
VIDEOS_DIR = os.path.join(BASE_DIR, "testing vidio")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Default Model Paths
PPE_MODEL_PATH = os.path.join(MODELS_DIR, "ppe_yolov8.pt")
DEFAULT_YOLO_MODEL = "yolov8n.pt"

# PPE Detection Classes (Standard PPE Dataset Schema)
PPE_CLASSES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}

# Color palette (BGR for OpenCV, RGB/Hex for Streamlit)
COLORS_BGR = {
    "compliant": (40, 167, 69),      # Green
    "violation": (0, 0, 220),        # Red
    "warning": (0, 165, 255),        # Orange
    "neutral": (220, 120, 0),        # Blue/Cyan
    "gear_vest": (0, 255, 255),      # Bright Yellow/Cyan
    "gear_helmet": (255, 255, 0),    # Cyan/Yellow
    "text": (255, 255, 255),         # White
    "dark_bg": (20, 20, 20)          # Near Black
}

# Default Compliance Rules
DEFAULT_RULES = {
    "require_hardhat": True,
    "require_vest": True,
    "require_mask": False,
    "confidence_threshold": 0.35,
    "iou_threshold": 0.45,
    "track_persistence": 30,  # frames
    "alert_cooldown_seconds": 3.0
}
