# Real-Time PPE Detection and Safety Compliance Monitoring Using Deep Learning

A production-ready Deep Learning & Computer Vision application for real-time Personal Protective Equipment (PPE) detection and safety compliance auditing on construction and industrial worksites.

Built with **Python**, **OpenCV**, **Ultralytics YOLOv8**, and **Streamlit**.

---

## 🌟 Key Features

1. **Multi-Class PPE Detection**:
   - Detects workers and essential safety gear:
     - ⛑️ **Hardhat / Safety Helmet** (and `No-Hardhat` violations)
     - 🦺 **High-Visibility Safety Vest** (and `No-Vest` violations)
     - 😷 **Face Mask** (optional toggle)
   - Supports dedicated pre-trained YOLOv8 PPE weights with automatic fallback to standard YOLOv8 with spatial computer vision heuristics.

2. **Compliance & Violation Engine**:
   - Evaluates compliance per worker in real-time.
   - **Green Bounding Boxes**: Fully compliant workers adhering to safety protocols.
   - **Red Bounding Boxes**: Violations with explicit tags (e.g., `Worker #1 | VIOLATION: NO HARDHAT, NO SAFETY VEST`).
   - Spatial association (head region analysis for helmets, torso region analysis for safety vests).

3. **Multi-Worker Tracking & Temporal Smoothing**:
   - IoU and centroid tracking persists worker identities (`Worker #1`, `Worker #2`).
   - Temporal smoothing over sliding frame windows prevents flickering false alarms when workers turn around or experience transient occlusion.

4. **Multi-Input Stream Dashboard (Streamlit)**:
   - **Workspace Sample Videos**: Instant 1-click testing of provided videos (`with saftey rule.mp4` and `violating safety rules.mp4`).
   - **Upload Video File**: Inspect `.mp4`, `.avi`, `.mov` footage.
   - **Upload Image**: Instant static audit of site photos.
   - **Live Webcam**: Real-time camera feed monitoring.

5. **Auditing & Reporting**:
   - Real-time KPI metrics (Total Detected, Compliant Count, Violations Count, Compliance Rate %).
   - Automated snapshot capture of every safety violation.
   - Detailed timestamped violation log table with direct **CSV export** for OSHA / site compliance audits.
   - Optional audio alert trigger upon detecting safety breaches.

---

## 🚀 Quick Start

### 1. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run the Streamlit Web Application
```powershell
.\.venv\Scripts\streamlit run app.py
```

### 3. Run Headless CLI Batch Verification Test
To verify the detector on the sample videos from the terminal:
```powershell
.\.venv\Scripts\python test_pipeline.py
```

---

## 📂 Project Structure

```
d:\dl project\
├── app.py                     # Streamlit web application dashboard
├── detector.py                # PPEDetector engine with YOLOv8 & CV heuristics
├── tracker.py                 # WorkerTracker with persistence & temporal smoothing
├── visualizer.py              # HUD banner & color-coded bounding box rendering
├── test_pipeline.py           # Verification script for batch testing
├── config.py                  # Colors, class schemas, default thresholds
├── requirements.txt           # Project dependencies
├── testing vidio/             # Test video datasets
│   ├── with saftey rule.mp4
│   └── violating safety rules.mp4
├── models/                    # Model weights directory
└── outputs/                   # Annotated videos and violation snapshots
    └── snapshots/
```
