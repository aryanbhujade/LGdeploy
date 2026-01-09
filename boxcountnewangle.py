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
import argparse

# For Python 3.8 compatibility with type hints
from typing import Tuple, Optional

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

#
# --- Configuration (camera-agnostic) ---
#
MODEL_VERSION = "4"

# Human-readable identifier for the stream/camera/area.
# Change per deployment (e.g., "DOCK19", "DOCK22", "CAMERA_53", etc.)
CAMERA_AREA = "CAMERA"

input_folder = "LG_videos"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Process any mp4 in the input folder by default
VIDEO_GLOB = "newangle.mp4"

# RTSP/video source can be provided via CLI args (see usage below).

# Robustness Knobs
MIN_TRACK_FRAMES_BEFORE_COUNT = 4
LINE_CROSS_EPS = 15.0  # Deadzone around the line
EVENT_COOLDOWN_SECONDS = 0.0  # legacy; kept for backward-compat but not used by the new logic
# Separate cooldowns per direction so a fast "tip over then pull back" reliably triggers -1.
IN_EVENT_COOLDOWN_SECONDS = 0.25
OUT_EVENT_COOLDOWN_SECONDS = 0.0
MIN_BOX_AREA = 800
ID_PURGE_FRAMES = 50
CONFIDENCE_THRESHOLD = 0.30

# Counting line definition (camera-agnostic)
# Horizontal line spanning the frame width, placed near the bottom.
LINE_OFFSET_FROM_BOTTOM_PX = 160

# --- Database (SQLite) ---
DB_PATH = "lg_counts.db"

# ---------------------------------------------------------------------------
# Usage Examples:
#   python boxcount.py --source "rtsp://..." --rtsp-test
#   python boxcount.py --source path/to/video.mp4
#   python boxcount.py --source "rtsp://..." --camera-area CAMERA_53
# ---------------------------------------------------------------------------

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
    """Signed value for which side of the oriented line a point lies on."""
    (lx1, ly1), (lx2, ly2) = p1, p2
    return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)


def bbox_rel_to_horizontal_line(x1: int, y1: int, x2: int, y2: int, line_y: int) -> Tuple[float, float, float]:
    """Return (top_rel, bottom_rel, bottom_center_y_rel) relative to a horizontal line at y=line_y.

    Negative means above the line, positive means below the line.
    We use bbox edges (top/bottom) instead of centroid because near the frame boundary
    the bbox can be clipped and centroid becomes unstable.
    """
    top_rel = float(y1 - line_y)
    bottom_rel = float(y2 - line_y)
    bc_rel = float(((y1 + y2) * 0.5) - line_y)
    return top_rel, bottom_rel, bc_rel


def classify_bbox_side(bottom_rel: float, eps: float) -> Optional[int]:
    """Classify which side the object is on using the bbox bottom edge.

    Returns:
      -1 => definitely above line
      +1 => definitely below line
      None => within deadzone
    """
    if bottom_rel <= -eps:
        return -1
    if bottom_rel >= eps:
        return 1
    return None

def build_count_line(curr_w: int, curr_h: int, offset_from_bottom_px: int):
    """Return a horizontal counting line spanning full width at (height - offset)."""
    y = max(0, int(curr_h) - int(offset_from_bottom_px))
    return (0, y), (int(curr_w) - 1, y)

# --- RTSP Quick Test Helper ---
def rtsp_quick_test(model, device, url, seconds=20, max_frames=None, sample_every_n=3):
    """Open an RTSP stream and run lightweight inference for a short window.

    This is meant as a connectivity + compatibility smoke-test on the LG admin machine.
    """
    print(f"\nRTSP test: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("RTSP failed: could not open")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"Opened: {w}x{h} reported_fps={fps:.2f}")

    t_start = time.perf_counter()
    frames = 0
    infer_frames = 0
    total_dets = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if (time.perf_counter() - t_start) < 5.0:
                continue
            print("RTSP failed: no frames")
            break

        frames += 1

        do_infer = (frames % max(1, int(sample_every_n)) == 0)
        if do_infer:
            infer_frames += 1
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                iou=0.70,
                max_det=300,
                device=device,
                verbose=False,
            )
            r = results[0]
            det_count = int(len(r.boxes)) if (r.boxes is not None) else 0
            total_dets += det_count

        elapsed = time.perf_counter() - t_start
        if max_frames is not None and frames >= int(max_frames):
            break
        if max_frames is None and elapsed >= float(seconds):
            break

    cap.release()

    elapsed = max(1e-6, time.perf_counter() - t_start)
    fps_measured = frames / elapsed
    infer_fps = infer_frames / elapsed
    avg_dets = (total_dets / max(1, infer_frames))

    print(
        f"RTSP ok: frames={frames} elapsed={elapsed:.2f}s fps={fps_measured:.2f} "
        f"infer_frames={infer_frames} infer_fps={infer_fps:.2f} avg_dets={avg_dets:.2f}"
    )


# --- CLI Argument Parser ---
def build_arg_parser():
    parser = argparse.ArgumentParser(description="LG Box Counter (video file or RTSP stream)")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Input source. Can be an RTSP URL (rtsp://...) or a path to a video file. "
            "If omitted, the script processes mp4s from input_folder/VIDEO_GLOB."
        ),
    )
    parser.add_argument(
        "--rtsp-test",
        action="store_true",
        help="Run a short RTSP smoke test (connectivity + lightweight inference) and exit.",
    )
    parser.add_argument("--rtsp-seconds", type=int, default=20, help="RTSP test duration in seconds")
    parser.add_argument(
        "--rtsp-max-frames",
        type=int,
        default=None,
        help="Optional cap on frames during RTSP test (overrides seconds if set)",
    )
    parser.add_argument(
        "--rtsp-sample-every",
        type=int,
        default=3,
        help="Run model on every Nth frame during RTSP test",
    )
    parser.add_argument(
        "--camera-area",
        type=str,
        default=None,
        help="Override CAMERA_AREA label used for logging/output naming",
    )
    parser.add_argument(
        "--line-offset",
        type=int,
        default=None,
        help="Override LINE_OFFSET_FROM_BOTTOM_PX",
    )
    return parser

# --- Main Execution ---
if __name__ == "__main__":
    print("LG Box Counter Starting...")
    init_db()

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.camera_area is not None:
        CAMERA_AREA = args.camera_area
    if args.line_offset is not None:
        LINE_OFFSET_FROM_BOTTOM_PX = int(args.line_offset)

    # Load Model (assume best.pt sits next to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(script_dir, "bestnew.pt")

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

    if args.rtsp_test:
        if args.source is None or not args.source.startswith("rtsp://"):
            raise ValueError("For --rtsp-test, you must provide --source with an RTSP URL (rtsp://...)")
        rtsp_quick_test(
            model=model,
            device=DEVICE,
            url=args.source,
            seconds=args.rtsp_seconds,
            max_frames=args.rtsp_max_frames,
            sample_every_n=args.rtsp_sample_every,
        )
        raise SystemExit(0)

    # Select video files or stream source
    if args.source is not None and not args.source.startswith("rtsp://"):
        video_files = [args.source]
    elif args.source is None:
        video_files = sorted(glob.glob(os.path.join(input_folder, VIDEO_GLOB)))
    elif args.source.startswith("rtsp://"):
        video_files = [args.source]
    else:
        video_files = []

    if not video_files:
        if args.source is None:
            print(f"No .mp4 files found in: {input_folder}")
            print("You can specify a source with --source path/to/video.mp4 or --source rtsp://...")
        else:
            print(f"No video files or stream found for source: {args.source}")
        raise SystemExit(1)

    for idx, video_path in enumerate(video_files):
        is_rtsp = video_path.startswith("rtsp://")
        video_name = "stream" if is_rtsp else os.path.splitext(os.path.basename(video_path))[0]
        # If user supplied --camera-area, use that for output naming
        area_tag = CAMERA_AREA
        # Generate Output Filename
        timestamp = datetime.now().strftime("%H%M_%d%m%Y")
        filename = f"{timestamp}_{area_tag}_{video_name}_LGmodelV{MODEL_VERSION}.mp4"
        output_path = os.path.join(output_folder, filename)

        print(f"\nProcessing: {video_name}.mp4" if not is_rtsp else f"\nProcessing RTSP stream: {video_path}")
        print(f"Saving to: {output_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video/stream: {video_path}")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if is_rtsp:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Build horizontal line near the bottom of the frame
        line_pt1, line_pt2 = build_count_line(frame_width, frame_height, LINE_OFFSET_FROM_BOTTOM_PX)
        print(f"Count line: {line_pt1} -> {line_pt2}")
        line_y = int(line_pt1[1])  # horizontal line

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
                # Robust position signals relative to the counting line
                top_rel, bottom_rel, bc_rel = bbox_rel_to_horizontal_line(x1b, y1b, x2b, y2b, line_y)

                if track_id not in track_state:
                    track_state[track_id] = {
                        "last_side": None,          # legacy (kept, but we now use stable_side)
                        "stable_side": None,        # -1 above, +1 below, None unknown/within deadzone
                        "last_in_time": -999.0,     # last time we incremented
                        "last_out_time": -999.0,    # last time we decremented
                        "last_seen_frame": frame_count,
                        "frames_seen": 0,
                        "counted": False,
                    }

                st = track_state[track_id]
                st["last_seen_frame"] = frame_count
                st["frames_seen"] += 1

                # Determine a stable side using bbox bottom edge (more stable near frame boundary)
                side_now = classify_bbox_side(bottom_rel, LINE_CROSS_EPS)
                prev_side = st.get("stable_side", None)

                # Initialize side if unknown (only when confidently away from line)
                if prev_side is None and side_now is not None:
                    st["stable_side"] = side_now
                    prev_side = side_now

                # Only consider transitions when we have a confident current side
                # Use overlap-based side logic for both increment and decrement, symmetrically.
                COUNTED_SIDE = 1  # BELOW the line is the counted side
                if side_now is not None and prev_side is not None and side_now != prev_side:
                    # Increment: moved ONTO counted side (bottom edge goes from above->below)
                    if side_now == COUNTED_SIDE and prev_side != COUNTED_SIDE:
                        # For IN we still want a tiny amount of protection against instant re-ids/jitter,
                        # and we only count once the track has existed a few frames.
                        if st["frames_seen"] > MIN_TRACK_FRAMES_BEFORE_COUNT:
                            if (current_time - st["last_in_time"]) > IN_EVENT_COOLDOWN_SECONDS:
                                boxes_loaded += 1
                                st["counted"] = True
                                st["last_in_time"] = current_time
                                print(f"ID {int(track_id)} IN -> Total: {boxes_loaded}")
                                log_count(CAMERA_AREA, boxes_loaded, +1, video_name, WEIGHTS_PATH, MODEL_VERSION)

                    # Decrement: moved OFF counted side (bottom edge goes from below->above)
                    elif prev_side == COUNTED_SIDE and side_now != COUNTED_SIDE:
                        # For OUT: allow fast "tip over then pull back" to immediately -1.
                        # Only require that this track had been counted.
                        if st.get("counted", False):
                            if (current_time - st["last_out_time"]) > OUT_EVENT_COOLDOWN_SECONDS:
                                boxes_loaded = max(0, boxes_loaded - 1)
                                st["counted"] = False
                                st["last_out_time"] = current_time
                                print(f"ID {int(track_id)} OUT -> Total: {boxes_loaded}")
                                log_count(CAMERA_AREA, boxes_loaded, -1, video_name, WEIGHTS_PATH, MODEL_VERSION)

                    # Update stable side after processing
                    st["stable_side"] = side_now

                # Keep legacy last_side updated for debugging/compat
                st["last_side"] = 1.0 if side_now == 1 else (-1.0 if side_now == -1 else st.get("last_side"))

                color = (0, 255, 0) if st.get("counted", False) else (0, 0, 255)
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 2)
                cv2.putText(frame, f"ID {int(track_id)}", (x1b, y1b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ids_to_remove = [tid for tid, st in track_state.items() if (frame_count - st["last_seen_frame"]) > ID_PURGE_FRAMES]
            for tid in ids_to_remove:
                del track_state[tid]

            cv2.line(frame, line_pt1, line_pt2, (0, 0, 255), 2)
            cv2.putText(frame, f"Count: {boxes_loaded}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 4)
            out.write(frame)
            cv2.imshow("LG Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()

        log_count(CAMERA_AREA, boxes_loaded, 0, video_name + "_FINAL", WEIGHTS_PATH, MODEL_VERSION)
        dt = time.perf_counter() - t0
        fps_val = frame_count / dt if dt > 0 else 0.0
        print(f"Processed {frame_count} frames in {dt:.2f}s -> {fps_val:.2f} FPS")
        print(f"Done! Saved to: {output_path}")

    cv2.destroyAllWindows()