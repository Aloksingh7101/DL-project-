"""
Verification and CLI batch testing script for Real-Time PPE Detection & Safety Compliance.
Tests both provided videos ('with saftey rule.mp4' and 'violating safety rules.mp4')
and logs detection accuracy and compliance summary.
"""
import os
import cv2
from detector import PPEDetector
from tracker import WorkerTracker
from visualizer import draw_hud_banner, draw_worker_annotations
from config import VIDEOS_DIR, OUTPUTS_DIR

def run_test_on_video(video_filename, max_frames=60):
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist.")
        return

    print(f"\n==========================================")
    print(f"Testing Video: {video_filename}")
    print(f"==========================================")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    detector = PPEDetector(conf_threshold=0.35)
    tracker = WorkerTracker()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total Frames: {total_frames}, FPS: {fps:.2f}")

    frame_idx = 0
    compliant_events = 0
    violation_events = 0

    # Output test video
    out_path = os.path.join(OUTPUTS_DIR, f"annotated_{video_filename}")
    writer = None

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Downscale for faster test processing
        if frame.shape[0] > 1080:
            scale = 1080.0 / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        workers = detector.detect(frame)
        tracked = tracker.update(workers)

        active_compliant = sum(1 for w in tracked if w['is_compliant'])
        active_violation = len(tracked) - active_compliant

        compliant_events += active_compliant
        violation_events += active_violation

        for w in tracked:
            draw_worker_annotations(frame, w, show_gear_boxes=True)

        frame = draw_hud_banner(frame, len(tracked), active_compliant, active_violation)
        writer.write(frame)

        if frame_idx % 15 == 0:
            print(f"Frame {frame_idx}/{max_frames} -> Tracked Workers: {len(tracked)}, Compliant: {active_compliant}, Violations: {active_violation}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved sample annotated video to: {out_path}")

    print(f"Summary for {video_filename}:")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Compliant detections: {compliant_events}")
    print(f"  Violation detections: {violation_events}")

if __name__ == "__main__":
    for f in ["with saftey rule.mp4", "violating safety rules.mp4"]:
        run_test_on_video(f, max_frames=50)
