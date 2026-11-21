"""
Streamlit Web Application for Real-Time PPE Detection & Safety Compliance Monitoring
"""
import os
import sys
import time
import tempfile
from datetime import datetime
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

# Local modules
from config import BASE_DIR, VIDEOS_DIR, SNAPSHOTS_DIR, OUTPUTS_DIR, DEFAULT_RULES, LOGO_PATH
from detector import PPEDetector
from tracker import WorkerTracker
from visualizer import draw_hud_banner, draw_worker_annotations

# Page configuration
st.set_page_config(
    page_title="PPE Detection & Safety Compliance Monitoring",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern industrial safety theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .val-green { color: #16A34A; }
    .val-red { color: #DC2626; }
    .val-blue { color: #2563EB; }
    .val-amber { color: #D97706; }
    .badge-safe {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-violation {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "violation_logs" not in st.session_state:
    st.session_state.violation_logs = []
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "total_seen": 0,
        "compliant_seen": 0,
        "violation_seen": 0,
        "active_ids": set()
    }
if "alert_history" not in st.session_state:
    st.session_state.alert_history = set()  # set of (track_id, missing_gear_tuple)

@st.cache_resource
def get_detector(conf_thresh, iou_thresh):
    """Cache the PPE Detector model in memory."""
    return PPEDetector(conf_threshold=conf_thresh, iou_threshold=iou_thresh)

def log_violation(track_id, missing_gear, frame, box):
    """Log a unique violation event and save snapshot."""
    gear_key = tuple(sorted(missing_gear))
    event_key = (track_id, gear_key)
    
    # Avoid duplicate logging of same violation for same worker in quick succession
    if event_key not in st.session_state.alert_history:
        st.session_state.alert_history.add(event_key)
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_time = now.strftime("%Y%m%d_%H%M%S_%f")
        
        # Save snapshot crop
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        
        snapshot_filename = f"violation_worker_{track_id}_{file_time}.jpg"
        snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
        if crop.size > 0:
            cv2.imwrite(snapshot_path, crop)
        
        log_entry = {
            "Timestamp": timestamp_str,
            "Worker ID": f"Worker #{track_id}",
            "Status": "VIOLATION",
            "Missing Gear": ", ".join(missing_gear) if missing_gear else "Unspecified Breach",
            "Snapshot": snapshot_filename
        }
        st.session_state.violation_logs.insert(0, log_entry)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=76)
    else:
        st.markdown("<div style='font-size: 3rem;'>🦺</div>", unsafe_allow_html=True)
    st.title("Safety Control Center")
    
    st.subheader("1. Input Source")
    input_source = st.radio(
        "Select Video / Image Stream",
        ["Sample Videos (In Workspace)", "Upload Video File", "Upload Image", "Live Webcam"],
        index=0
    )

    st.subheader("2. Compliance Rules")
    req_hardhat = st.checkbox("Require Hardhat / Helmet", value=True)
    req_vest = st.checkbox("Require Safety Vest", value=True)
    req_mask = st.checkbox("Require Face Mask", value=False)
    
    active_rules = {
        "require_hardhat": req_hardhat,
        "require_vest": req_vest,
        "require_mask": req_mask
    }

    st.subheader("3. Detection Sensitivity")
    conf_thresh = st.slider("Confidence Threshold", min_value=0.15, max_value=0.90, value=0.35, step=0.05)
    iou_thresh = st.slider("NMS IoU Threshold", min_value=0.20, max_value=0.70, value=0.45, step=0.05)
    show_gear_details = st.checkbox("Show Gear Bounding Boxes", value=True)
    enable_audio = st.checkbox("Trigger Audio Alert on Violation", value=False)

    st.divider()
    st.subheader("📊 Live Session Metrics")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Total Detected", st.session_state.session_stats["total_seen"])
        st.metric("Violations", st.session_state.session_stats["violation_seen"], delta=None)
    with m_col2:
        st.metric("Compliant", st.session_state.session_stats["compliant_seen"])
        total_s = st.session_state.session_stats["total_seen"]
        comp_s = st.session_state.session_stats["compliant_seen"]
        rate_s = int((comp_s / total_s * 100)) if total_s > 0 else 100
        st.metric("Compliance Rate", f"{rate_s}%")

    if st.button("🔄 Reset Session Metrics", use_container_width=True):
        st.session_state.violation_logs = []
        st.session_state.alert_history = set()
        st.session_state.session_stats = {
            "total_seen": 0,
            "compliant_seen": 0,
            "violation_seen": 0,
            "active_ids": set()
        }
        st.rerun()

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-header">🛡️ Real-Time PPE Detection & Safety Compliance Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Deep Learning & Computer Vision Inspection for Construction & Industrial Worksites</div>', unsafe_allow_html=True)

# Load detector
detector = get_detector(conf_thresh, iou_thresh)
tracker = WorkerTracker(max_missed=25, iou_thresh=0.25)

# Hidden HTML5 audio element for safety breach buzzer
if enable_audio:
    st.markdown("""
    <audio id="alert-audio" preload="auto">
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
    </audio>
    <script>
        function playAlert() {
            var audio = document.getElementById('alert-audio');
            if (audio) { audio.currentTime = 0; audio.play().catch(function(){}); }
        }
    </script>
    """, unsafe_allow_html=True)

# Tab layout for Visual Monitoring vs Violation Audit Log
tab_monitor, tab_logs = st.tabs(["🎥 Live Monitoring Feed", "📋 Violation Audit Log & Reports"])

with tab_monitor:
    video_source_path = None
    image_file = None
    use_webcam = False

    if input_source == "Sample Videos (In Workspace)":
        sample_options = []
        if os.path.exists(VIDEOS_DIR):
            sample_options = [f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".avi", ".mov"))]
        
        if sample_options:
            selected_sample = st.selectbox("Choose Sample Test Video:", sample_options, index=0)
            video_source_path = os.path.join(VIDEOS_DIR, selected_sample)
            st.info(f"Loaded workspace video: `{selected_sample}`")
        else:
            st.warning("No sample videos found in `testing vidio` directory.")

    elif input_source == "Upload Video File":
        uploaded_video = st.file_uploader("Upload a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_source_path = tfile.name
            st.success(f"Video uploaded: {uploaded_video.name}")

    elif input_source == "Upload Image":
        image_file = st.file_uploader("Upload an image file (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"])

    elif input_source == "Live Webcam":
        use_webcam = True
        st.info("Webcam stream selected. Click 'Start Stream' below to begin live processing.")

    # ----------------- IMAGE PROCESSING -----------------
    if image_file is not None:
        pil_img = Image.open(image_file).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Run detection
        workers = detector.detect(frame, rules=active_rules)
        tracked_workers = tracker.update(workers)
        
        compliant_n = sum(1 for w in tracked_workers if w['is_compliant'])
        violation_n = len(tracked_workers) - compliant_n

        for w in tracked_workers:
            draw_worker_annotations(frame, w, show_gear_boxes=show_gear_details)
            if not w['is_compliant']:
                log_violation(w.get('track_id', 1), w.get('missing_gear', []), frame, w['box'])

        frame = draw_hud_banner(frame, len(tracked_workers), compliant_n, violation_n)
        annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        col_img1, col_img2 = st.columns([3, 1])
        with col_img1:
            st.image(annotated_rgb, caption="Annotated Safety Compliance Result", use_container_width=True)
        with col_img2:
            st.subheader("Inspection Summary")
            st.metric("Workers Detected", len(tracked_workers))
            st.metric("Compliant", compliant_n)
            st.metric("Violations", violation_n)

    # ----------------- VIDEO / WEBCAM PROCESSING -----------------
    elif video_source_path is not None or use_webcam:
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            start_btn = st.button("▶️ Start Monitoring", key="btn_start", use_container_width=True, type="primary")
        with c2:
            stop_btn = st.button("⏹️ Stop Monitoring", key="btn_stop", use_container_width=True)

        video_placeholder = st.empty()
        status_banner_placeholder = st.empty()

        if start_btn:
            cap = None
            if use_webcam:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(video_source_path)

            if not cap.isOpened():
                st.error("Failed to open video source or camera.")
            else:
                fps_time = time.time()
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    # Skip alternate frames if needed for smooth playback speed on large 4K files
                    if frame.shape[0] > 1920:
                        # Downscale 4K to 1080p for real-time high-speed inference
                        scale_factor = 1080.0 / frame.shape[1]
                        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

                    # Detection and tracking
                    workers = detector.detect(frame, rules=active_rules)
                    tracked_workers = tracker.update(workers)

                    active_compliant = 0
                    active_violations = 0

                    for w in tracked_workers:
                        tid = w.get('track_id', 1)
                        if tid not in st.session_state.session_stats["active_ids"]:
                            st.session_state.session_stats["active_ids"].add(tid)
                            st.session_state.session_stats["total_seen"] += 1

                        if w['is_compliant']:
                            active_compliant += 1
                        else:
                            active_violations += 1
                            log_violation(tid, w.get('missing_gear', []), frame, w['box'])

                        draw_worker_annotations(frame, w, show_gear_boxes=show_gear_details)

                    # Update cumulative stats
                    st.session_state.session_stats["violation_seen"] = len(st.session_state.alert_history)
                    st.session_state.session_stats["compliant_seen"] = max(
                        0, st.session_state.session_stats["total_seen"] - st.session_state.session_stats["violation_seen"]
                    )

                    frame = draw_hud_banner(frame, len(tracked_workers), active_compliant, active_violations)

                    # FPS calculation
                    curr_time = time.time()
                    fps = 1.0 / (curr_time - fps_time) if (curr_time - fps_time) > 0 else 30.0
                    fps_time = curr_time

                    # Render frame in Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    if active_violations > 0:
                        status_banner_placeholder.error(f"⚠️ ACTIVE SAFETY VIOLATION DETECTED: {active_violations} worker(s) missing required PPE!")
                    else:
                        status_banner_placeholder.success("✅ Worksite Status: 100% PPE Safety Compliant")

                    # Break if stopped
                    if stop_btn:
                        break

                cap.release()
                st.info("Video processing completed.")

with tab_logs:
    st.subheader("📋 Safety Compliance Violation Records")
    
    if st.session_state.violation_logs:
        df = pd.DataFrame(st.session_state.violation_logs)
        
        # Display Download Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Violation Log (CSV)",
            data=csv_data,
            file_name=f"ppe_violations_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Display Dataframe
        st.dataframe(df[["Timestamp", "Worker ID", "Status", "Missing Gear"]], use_container_width=True)

        st.subheader("📸 Captured Violation Snapshots")
        cols = st.columns(4)
        for idx, entry in enumerate(st.session_state.violation_logs[:8]):
            snap_path = os.path.join(SNAPSHOTS_DIR, entry["Snapshot"])
            if os.path.exists(snap_path):
                with cols[idx % 4]:
                    st.image(snap_path, caption=f"{entry['Worker ID']} - {entry['Missing Gear']}", use_container_width=True)
    else:
        st.info("No safety violations recorded yet. Active monitoring log is clean!")
