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

output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# NOTE: On the LG server we always run with an explicit --source (RTSP URL or video file path).
# The script no longer scans an input folder.

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

# --- Stats printing interval ---
STATS_PRINT_EVERY_S = 20

# --- Database (SQLite) ---
DB_PATH = "lg_counts.db"

# We now log ONE row per saved output file (mp4 or RTSP clip), not per event.
# This keeps the DB compact and aligned with video artifacts.

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS box_count_clips (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              camera_area TEXT NOT NULL,
              clip_start_utc TEXT NOT NULL,
              clip_end_utc TEXT NOT NULL,
              source TEXT NOT NULL,
              output_path TEXT NOT NULL,
              final_box_count INTEGER NOT NULL,
              model_path TEXT,
              model_version TEXT
            )
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_box_count_clips_area_time
            ON box_count_clips(camera_area, clip_start_utc)
            """
        )


def log_clip_summary(
    camera_area: str,
    clip_start_utc: str,
    clip_end_utc: str,
    source: str,
    output_path: str,
    final_box_count: int,
    model_path: str,
    model_version: str,
) -> None:
    """Log one record per saved clip/video."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT INTO box_count_clips(
              camera_area, clip_start_utc, clip_end_utc,
              source, output_path, final_box_count,
              model_path, model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(camera_area),
                str(clip_start_utc),
                str(clip_end_utc),
                str(source),
                str(output_path),
                int(final_box_count),
                str(model_path) if model_path is not None else None,
                str(model_version) if model_version is not None else None,
            ),
        )

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

# --- CLI Argument Parser ---
def build_arg_parser():
    parser = argparse.ArgumentParser(description="LG Box Counter (video file or RTSP stream)")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Input source. Can be an RTSP URL (rtsp://...) or a path to a video file. "
            "This argument is required on the LG server."
        ),
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
    # Removed --stats-every argument, now hardcoded
    parser.add_argument(
        "--clip-time",
        type=int,
        nargs="?",
        const=600,
        default=600,
        help=(
            "For RTSP sources, rotate/split the output into clips of this duration in seconds. "
            "If you pass '--clip-time' without a value, it uses 600. Ignored for normal video files."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live display window (cv2.imshow). Useful on servers.",
    )
    return parser

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_output_path(output_folder: str, area_tag: str, video_name: str, model_version: str, clip_index: int = 0) -> str:
    timestamp = datetime.now().strftime("%H%M_%d%m%Y")
    if clip_index > 0:
        filename = f"{timestamp}_{area_tag}_{video_name}_clip{clip_index:03d}_LGmodelV{model_version}.mp4"
    else:
        filename = f"{timestamp}_{area_tag}_{video_name}_LGmodelV{model_version}.mp4"
    return os.path.join(output_folder, filename)

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

    # Always process exactly the provided source (RTSP URL or file path)
    video_path = args.source
    is_rtsp = video_path.startswith("rtsp://")
    video_name = "stream" if is_rtsp else os.path.splitext(os.path.basename(video_path))[0]

    # If user supplied --camera-area, use that for output naming
    area_tag = CAMERA_AREA

    # Output path: for RTSP we will rotate into clips; for files it's one output.
    clip_index = 0
    output_path = make_output_path(output_folder, area_tag, video_name, MODEL_VERSION, clip_index=0)

    print(f"\nProcessing: {video_name}.mp4" if not is_rtsp else f"\nProcessing RTSP stream: {video_path}")
    if is_rtsp:
        print(f"RTSP clip rotation: {int(args.clip_time)}s per clip")
    print(f"Saving to: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video/stream: {video_path}")
        raise SystemExit(1)

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

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Clip bookkeeping (RTSP only)
    clip_start_wall = time.perf_counter()
    clip_start_utc = utc_now_iso()
    clip_frames = 0

    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

    track_state = {}
    boxes_loaded = 0
    t0 = time.perf_counter()
    frame_count = 0
    dt_per_frame = 1.0 / fps
    stats_last_print_t = time.perf_counter()
    stats_every_s = int(STATS_PRINT_EVERY_S)
    # For non-RTSP, ensure clip_start_utc is set
    if not is_rtsp:
        clip_start_utc = utc_now_iso()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        clip_frames += 1
        current_time = frame_count * dt_per_frame

        now_t = time.perf_counter()
        if (now_t - stats_last_print_t) >= float(stats_every_s):
            elapsed = max(1e-6, now_t - t0)
            proc_fps = frame_count / elapsed
            rt_pct = (proc_fps / max(1e-6, float(fps))) * 100.0
            print(f"Perf: frames={frame_count} elapsed={elapsed:.1f}s avg_fps={proc_fps:.2f} realtime={rt_pct:.1f}%")
            stats_last_print_t = now_t

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
                            # DB logging per event removed

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
                            # DB logging per event removed

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
        if not args.no_display:
            cv2.imshow("LG Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # RTSP: rotate output into fixed-duration clips
        if is_rtsp:
            if (time.perf_counter() - clip_start_wall) >= float(args.clip_time):
                # finalize current clip
                out.release()
                clip_end_utc = utc_now_iso()
                try:
                    log_clip_summary(
                        camera_area=CAMERA_AREA,
                        clip_start_utc=clip_start_utc,
                        clip_end_utc=clip_end_utc,
                        source=video_path,
                        output_path=output_path,
                        final_box_count=boxes_loaded,
                        model_path=WEIGHTS_PATH,
                        model_version=MODEL_VERSION,
                    )
                except Exception as e:
                    print(f"DB log failed (clip summary): {e}")

                # start a new clip: reset RTSP stream and all per-clip state
                clip_index += 1
                output_path = make_output_path(output_folder, area_tag, video_name, MODEL_VERSION, clip_index=clip_index)
                print(f"Rotating clip -> {output_path}")
                cap.release()
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not reopen RTSP stream: {video_path}")
                    break
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                line_pt1, line_pt2 = build_count_line(frame_width, frame_height, LINE_OFFSET_FROM_BOTTOM_PX)
                line_y = int(line_pt1[1])
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
                tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
                track_state = {}
                clip_start_wall = time.perf_counter()
                clip_start_utc = utc_now_iso()
                clip_frames = 0

    cap.release()
    out.release()

    # Finalize DB logging once per saved output
    clip_end_utc = utc_now_iso()
    if is_rtsp:
        try:
            log_clip_summary(
                camera_area=CAMERA_AREA,
                clip_start_utc=clip_start_utc,
                clip_end_utc=clip_end_utc,
                source=video_path,
                output_path=output_path,
                final_box_count=boxes_loaded,
                model_path=WEIGHTS_PATH,
                model_version=MODEL_VERSION,
            )
        except Exception as e:
            print(f"DB log failed (final RTSP clip): {e}")
    else:
        try:
            log_clip_summary(
                camera_area=CAMERA_AREA,
                clip_start_utc=clip_start_utc,
                clip_end_utc=clip_end_utc,
                source=video_path,
                output_path=output_path,
                final_box_count=boxes_loaded,
                model_path=WEIGHTS_PATH,
                model_version=MODEL_VERSION,
            )
        except Exception as e:
            print(f"DB log failed (video summary): {e}")

    dt = time.perf_counter() - t0
    fps_val = frame_count / dt if dt > 0 else 0.0
    print(f"Processed {frame_count} frames in {dt:.2f}s -> {fps_val:.2f} FPS")
    print(f"Done! Saved to: {output_path}")

    cv2.destroyAllWindows()