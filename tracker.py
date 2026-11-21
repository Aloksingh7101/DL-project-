"""
Worker Tracking and Violation Event Manager
Maintains consistent identities for workers across frames and temporal smoothing for compliance states.
"""
import time
from collections import deque
import numpy as np

def calculate_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom

class TrackedWorker:
    def __init__(self, track_id, box, initial_status, missing_gear):
        self.track_id = track_id
        self.box = box
        self.centroid = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.missed_frames = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_alert_time = 0.0
        
        # History queue for temporal smoothing
        self.history_len = 7
        self.compliance_history = deque([initial_status], maxlen=self.history_len)
        self.missing_gear = missing_gear
        self.is_compliant = initial_status

    def update(self, box, is_compliant, missing_gear):
        self.box = box
        self.centroid = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.missed_frames = 0
        self.last_seen = time.time()
        self.compliance_history.append(is_compliant)
        self.missing_gear = missing_gear
        
        # Temporal smoothing: majority vote over recent frames
        compliant_votes = sum(1 for status in self.compliance_history if status)
        self.is_compliant = (compliant_votes / len(self.compliance_history)) >= 0.5

class WorkerTracker:
    def __init__(self, max_missed=25, iou_thresh=0.25):
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh
        self.next_id = 1
        self.workers = {}  # track_id -> TrackedWorker

    def update(self, detected_workers):
        """
        detected_workers: list of dicts with keys:
            'box': [x1, y1, x2, y2],
            'is_compliant': bool,
            'missing_gear': list of str,
            'gear_boxes': list of dict
        Returns list of matched dicts enriched with 'track_id' and smoothed 'is_compliant'.
        """
        current_tracked = []
        if not self.workers:
            # First frame or no active workers
            for det in detected_workers:
                worker = TrackedWorker(
                    track_id=self.next_id,
                    box=det['box'],
                    initial_status=det['is_compliant'],
                    missing_gear=det['missing_gear']
                )
                self.workers[self.next_id] = worker
                det['track_id'] = self.next_id
                det['is_compliant'] = worker.is_compliant
                self.next_id += 1
                current_tracked.append(det)
            return current_tracked

        # Match existing workers with new detections using IoU / Centroid distance
        existing_ids = list(self.workers.keys())
        existing_boxes = [self.workers[tid].box for tid in existing_ids]

        matched_detections = set()
        matched_workers = set()

        if len(detected_workers) > 0 and len(existing_ids) > 0:
            iou_matrix = np.zeros((len(existing_ids), len(detected_workers)), dtype=np.float32)
            for i, e_box in enumerate(existing_boxes):
                for j, det in enumerate(detected_workers):
                    iou_matrix[i, j] = calculate_iou(e_box, det['box'])

            # Greedy bipartite matching
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_thresh:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = existing_ids[i]
                self.workers[tid].update(
                    detected_workers[j]['box'],
                    detected_workers[j]['is_compliant'],
                    detected_workers[j]['missing_gear']
                )
                detected_workers[j]['track_id'] = tid
                detected_workers[j]['is_compliant'] = self.workers[tid].is_compliant
                matched_workers.add(tid)
                matched_detections.add(j)
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Unmatched detections become new workers
        for j, det in enumerate(detected_workers):
            if j not in matched_detections:
                worker = TrackedWorker(
                    track_id=self.next_id,
                    box=det['box'],
                    initial_status=det['is_compliant'],
                    missing_gear=det['missing_gear']
                )
                self.workers[self.next_id] = worker
                det['track_id'] = self.next_id
                det['is_compliant'] = worker.is_compliant
                self.next_id += 1

        # Mark missed frames for unmatched existing workers
        to_delete = []
        for tid, worker in self.workers.items():
            if tid not in matched_workers:
                worker.missed_frames += 1
                if worker.missed_frames > self.max_missed:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.workers[tid]

        return detected_workers
