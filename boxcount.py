import os
import glob
import warnings
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import sqlite3

# Suppress annoying pytorch warnings
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# ===========================================================================
# PART 1: THE SORT TRACKER (Streamlined)
# ===========================================================================


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            x, y = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(x, y)))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ===========================================================================
# PART 2: THE INFERENCE & COUNTING LOGIC
# ===========================================================================

# --- Configuration (DOCK19 ONLY) ---
MODEL_VERSION = "4"
DOCK_NAME = "DOCK19"

input_folder = "LG_videos"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Only process DOCK19 videos (including Decount variants)
VIDEO_GLOB = "DOCK19*.mp4"

# Robustness Knobs
MIN_TRACK_FRAMES_BEFORE_COUNT = 4
LINE_CROSS_EPS = 15.0  # Deadzone around the line
EVENT_COOLDOWN_SECONDS = 1.5
MIN_BOX_AREA = 800
ID_PURGE_FRAMES = 50
CONFIDENCE_THRESHOLD = 0.30

# Base Resolution (the coordinates below were calibrated on this resolution)
BASE_RES_W = 2592.0
BASE_RES_H = 1944.0

# DOCK19 line coordinates (defined at BASE_RES)
DOCK19_LINE_BASE = ((1523, 718), (2202, 874))

# --- Database (SQLite) ---
DB_PATH = "lg_counts.db"

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS box_counts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          camera_area TEXT NOT NULL,
          datetime_utc TEXT NOT NULL,
          box_count INTEGER NOT NULL,
          delta INTEGER NOT NULL,
          source TEXT NOT NULL,
          model_path TEXT,
          model_version TEXT
        )
        """)
        con.execute("""
        CREATE INDEX IF NOT EXISTS idx_box_counts_area_time
        ON box_counts(camera_area, datetime_utc)
        """)

def log_count(camera_area, box_count, delta, source, model_path, model_version):
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO box_counts(camera_area, datetime_utc, box_count, delta, source, model_path, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (camera_area, ts, box_count, delta, source, model_path, model_version))

# --- Helper Functions ---

def point_side(px: float, py: float, p1, p2) -> float:
    (lx1, ly1), (lx2, ly2) = p1, p2
    return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)

def scale_line(pt1, pt2, curr_w, curr_h):
    scale_x = curr_w / BASE_RES_W
    scale_y = curr_h / BASE_RES_H
    return (int(pt1[0] * scale_x), int(pt1[1] * scale_y)), (int(pt2[0] * scale_x), int(pt2[1] * scale_y))

# --- Main Execution ---
if __name__ == "__main__":
    print("LG Box Counter Starting...")
    init_db()
    
    # Load Model (assume best.pt sits next to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(script_dir, "best.pt")

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Could not find model weights 'best.pt' next to the script.\n"
            f"Expected: {WEIGHTS_PATH}\n"
            f"Place best.pt in the same folder as boxcount.py (or update WEIGHTS_PATH)."
        )
    print(f"Using weights: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    # Device Check (LG Windows Admin Machine: NVIDIA CUDA)
    try:
        DEVICE = 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"
    print(f"Inference device: {DEVICE}")

    # Optional: Print GPU name for sanity on LG machine
    if DEVICE != "cpu":
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # Process Videos
    video_files = sorted(glob.glob(os.path.join(input_folder, VIDEO_GLOB)))
    
    if not video_files:
        print(f"No .mp4 files found in: {input_folder}")
    
    for idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # DOCK19-only: always use the DOCK19 line
        base_pt1, base_pt2 = DOCK19_LINE_BASE

        # Generate Output Filename
        timestamp = datetime.now().strftime("%H%M_%d%m%Y")
        filename = f"{timestamp}_{DOCK_NAME}_{video_name}_LGmodelV{MODEL_VERSION}.mp4"
        output_path = os.path.join(output_folder, filename)

        print(f"\nProcessing: {video_name}.mp4")
        print(f"Saving to: {output_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Auto-Scale Line
        line_pt1, line_pt2 = scale_line(base_pt1, base_pt2, frame_width, frame_height)
        print(f"Scaled line: {line_pt1} -> {line_pt2}")

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
        
        track_state = {}
        boxes_loaded = 0
        t0 = time.perf_counter()
        frame_count = 0
        dt_per_frame = 1.0 / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            current_time = frame_count * dt_per_frame

            # Inference
            results = model.predict(source=frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, 
                                    iou=0.70, max_det=300, device=DEVICE, verbose=False)
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                detections = np.empty((0, 5))
            else:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
                wh = xyxy[:, 2:4] - xyxy[:, 0:2]
                areas = (wh[:, 0] * wh[:, 1])
                keep = (areas >= MIN_BOX_AREA)
                detections = np.concatenate([xyxy[keep], conf[keep]], axis=1) if np.any(keep) else np.empty((0, 5))

            tracked_objects = tracker.update(detections)

            # Counting Logic
            for *xyxy, track_id in tracked_objects:
                x1b, y1b, x2b, y2b = map(int, xyxy)
                cx, cy = int((x1b + x2b) / 2), int((y1b + y2b) / 2)

                if track_id not in track_state:
                    track_state[track_id] = {
                        "last_side": None, 
                        "last_event_time": -999.0, 
                        "last_seen_frame": frame_count, 
                        "frames_seen": 0
                    }
                
                st = track_state[track_id]
                st["last_seen_frame"] = frame_count
                st["frames_seen"] += 1

                current_side_val = point_side(cx, cy, line_pt1, line_pt2)

                if abs(current_side_val) >= LINE_CROSS_EPS:
                    prev_side_val = st["last_side"]
                    
                    if prev_side_val is None:
                        st["last_side"] = current_side_val
                    else:
                        if (prev_side_val < 0 and current_side_val > 0) or (prev_side_val > 0 and current_side_val < 0):
                            if (current_time - st["last_event_time"]) > EVENT_COOLDOWN_SECONDS:
                                if prev_side_val > 0 and current_side_val < 0:
                                    if st["frames_seen"] > MIN_TRACK_FRAMES_BEFORE_COUNT:
                                        boxes_loaded += 1
                                        st["last_event_time"] = current_time
                                        print(f"ID {int(track_id)} IN -> Total: {boxes_loaded}")
                                        log_count(DOCK_NAME, boxes_loaded, +1, video_name, WEIGHTS_PATH, MODEL_VERSION)
                                elif prev_side_val < 0 and current_side_val > 0:
                                    if st["frames_seen"] > MIN_TRACK_FRAMES_BEFORE_COUNT:
                                        boxes_loaded -= 1
                                        st["last_event_time"] = current_time
                                        print(f"ID {int(track_id)} OUT -> Total: {boxes_loaded}")
                                        log_count(DOCK_NAME, boxes_loaded, -1, video_name, WEIGHTS_PATH, MODEL_VERSION)
                        
                        st["last_side"] = current_side_val

                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
                cv2.putText(frame, f"ID {int(track_id)}", (x1b, y1b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            ids_to_remove = [tid for tid, st in track_state.items() if (frame_count - st["last_seen_frame"]) > ID_PURGE_FRAMES]
            for tid in ids_to_remove:
                del track_state[tid]

            cv2.line(frame, line_pt1, line_pt2, (0, 0, 255), 2)
            cv2.putText(frame, f"Count: {boxes_loaded}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
            cv2.imshow("LG Counter - DOCK19", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        
        log_count(DOCK_NAME, boxes_loaded, 0, video_name + "_FINAL", WEIGHTS_PATH, MODEL_VERSION)
        dt = time.perf_counter() - t0
        fps_val = frame_count / dt if dt > 0 else 0.0
        print(f"Processed {frame_count} frames in {dt:.2f}s -> {fps_val:.2f} FPS")
        print(f"Done! Saved to: {output_path}")

    cv2.destroyAllWindows()