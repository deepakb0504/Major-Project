"""
Analytics engine: process video with YOLO + built-in BoT-SORT tracking,
risk analysis (velocity/accel -> SUDDEN_ACCEL), color by risk (green->red),
output video, heatmap, and dwell-time analytics.
"""
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque

OUTPUT_DIR = Path(__file__).resolve().parent / "smartshop_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
MODEL_PATH_LOCAL = MODELS_DIR / "yolov8n.pt"
MODELS_DIR.mkdir(exist_ok=True)

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
HEATMAP_GRID_SIZE = 64
# Risk: acceleration threshold (pixels per frame^2)
ACCEL_THRESHOLD = 15.0
HISTORY_LEN = 8
RISK_DECAY_FRAMES = 30

# Risk 0-5 -> BGR (green = safe, red = high/theft)
RISK_BGR = [
    (94, 197, 34),   # 0 green safe
    (22, 204, 132),  # 1 lime
    (8, 179, 234),   # 2 yellow
    (16, 115, 249),  # 3 orange
    (68, 68, 239),   # 4 red
    (28, 28, 185),   # 5 dark red theft
]


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _risk_to_bgr(risk):
    r = max(0, min(5, int(risk)))
    return RISK_BGR[r]


def _draw_label(img, text, x, y, color_bgr=(255, 255, 255)):
    cv2.rectangle(img, (x, y - 18), (x + 8 * len(text) + 10, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)


def get_model_path():
    """
    Use YOLO model from backend/models/ so OneDrive doesn't lock the file.
    If only project-root yolov8n.pt exists, copy it once to backend/models/.
    """
    if MODEL_PATH_LOCAL.is_file():
        return str(MODEL_PATH_LOCAL)
    root_pt = BACKEND_DIR.parent / "yolov8n.pt"
    if root_pt.is_file():
        try:
            shutil.copy2(root_pt, MODEL_PATH_LOCAL)
            return str(MODEL_PATH_LOCAL)
        except (OSError, PermissionError) as e:
            raise PermissionError(
                "Permission denied reading yolov8n.pt (often due to OneDrive). "
                "Copy it manually: copy 'yolov8n.pt' into 'backend\\models\\yolov8n.pt' and try again."
            ) from e
    root_pt_zip = BACKEND_DIR.parent / "yolov8n.pt.zip"
    if root_pt_zip.is_file():
        try:
            shutil.copy2(root_pt_zip, MODEL_PATH_LOCAL)
            return str(MODEL_PATH_LOCAL)
        except (OSError, PermissionError) as e:
            raise PermissionError(
                "Permission denied creating backend\\models\\yolov8n.pt. "
                "If a folder exists at backend\\models\\yolov8n.pt, delete/rename it, then try again."
            ) from e
    return "yolov8n.pt"


def process_video(input_path: str, job_id: str):
    """
    Run person detection + YOLO built-in tracker (BoT-SORT) for accurate IDs and dwell time.
    Returns dict with output_video, heatmap paths, tracks, and summary.
    """
    try:
        model_path = get_model_path()
    except PermissionError as e:
        return {"error": str(e), "output_video": None, "heatmap": None, "tracks": None, "summary": None}
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        return {"error": str(e), "output_video": None, "heatmap": None, "tracks": None, "summary": None}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "Could not open video", "output_video": None, "heatmap": None, "tracks": None, "summary": None}

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    out_basename = f"output_{job_id[:8]}.mp4"
    out_path = OUTPUT_DIR / out_basename
    writer = None
    writer_kind = None
    try:
        import imageio
        writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        writer_kind = "imageio"
    except Exception:
        writer = None

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (max(w, 1), max(h, 1)))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (max(w, 1), max(h, 1)))
        writer_kind = "opencv"

    heatmap_grid = np.zeros((HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE), dtype=np.float32)
    frame_count = 0
    tracks_data = defaultdict(lambda: {"first_frame": None, "last_frame": None, "seen_frames": 0})
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    track_risk = defaultdict(lambda: {"score": 0, "decay_counter": 0})

    def _decay_risks():
        for tid, r in list(track_risk.items()):
            if r["decay_counter"] > 0:
                r["decay_counter"] -= 1
            else:
                r["score"] = max(0, r["score"] - 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if w == 0 or h == 0:
            h, w = frame.shape[:2]
            if writer_kind == "opencv":
                try:
                    writer.release()
                except Exception:
                    pass
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            iou=0.45,
        )
        annotated = frame.copy()
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            for i in range(len(ids)):
                tid = int(ids[i])
                x1, y1, x2, y2 = xyxy[i]
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                t = tracks_data[tid]
                if t["first_frame"] is None:
                    t["first_frame"] = frame_count
                t["last_frame"] = frame_count
                t["seen_frames"] += 1
                history = track_history[tid]
                risk_state = track_risk[tid]
                events = []
                if len(history) >= 2:
                    (_, cx0, cy0) = history[-1]
                    (_, cx1, cy1) = history[-2]
                    dt = 1.0
                    vx = (cx - cx0) / dt
                    vy = (cy - cy0) / dt
                    v_prev_x = (cx0 - cx1) / dt
                    v_prev_y = (cy0 - cy1) / dt
                    ax = (vx - v_prev_x) / dt
                    ay = (vy - v_prev_y) / dt
                    accel_mag = np.sqrt(ax * ax + ay * ay)
                    if accel_mag >= ACCEL_THRESHOLD:
                        events.append("SUDDEN_ACCEL")
                        risk_state["score"] = min(5, risk_state["score"] + 1)
                        risk_state["decay_counter"] = RISK_DECAY_FRAMES
                history.append((frame_count, cx, cy))
                risk = risk_state["score"]
                state = "NORMAL" if risk == 0 else "ALERT"
                dwell_s = t["seen_frames"] / float(fps)
                color_bgr = _risk_to_bgr(risk)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
                label_y = max(20, y1)
                _draw_label(annotated, f"ID {tid}", x1, label_y, color_bgr)
                _draw_label(annotated, state, x1, label_y + 18, color_bgr)
                _draw_label(annotated, f"Risk {risk}", x1, label_y + 36, color_bgr)
                _draw_label(annotated, f"{dwell_s:.1f}s", x1, label_y + 54, color_bgr)
                if events:
                    _draw_label(annotated, " ".join(events), x1, y2 + 18, color_bgr)
                cx_i = int((x1 + x2) / 2 * HEATMAP_GRID_SIZE / max(w, 1)) % HEATMAP_GRID_SIZE
                cy_i = int((y1 + y2) / 2 * HEATMAP_GRID_SIZE / max(h, 1)) % HEATMAP_GRID_SIZE
                heatmap_grid[cy_i, cx_i] += 1.0
                t["_max_risk"] = max(t.get("_max_risk", 0), risk)
                t["_events_set"] = t.get("_events_set", set()) | set(events)

        _decay_risks()
        if writer_kind == "imageio":
            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        else:
            writer.write(annotated)
        frame_count += 1

    cap.release()
    try:
        if writer_kind == "imageio":
            writer.close()
        else:
            writer.release()
    except Exception:
        pass

    # Heatmap: higher resolution, then resize for display
    heatmap_basename = f"heatmap_{job_id[:8]}.png"
    heatmap_path = OUTPUT_DIR / heatmap_basename
    if frame_count > 0 and heatmap_grid.max() > 0:
        hm_norm = (heatmap_grid / heatmap_grid.max() * 255).astype(np.uint8)
        hm_resized = cv2.resize(hm_norm, (256, 256), interpolation=cv2.INTER_LINEAR)
        hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        cv2.imwrite(str(heatmap_path), hm_colored)
    else:
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(heatmap_path), blank)

    # Build tracking analytics with risk/state/events
    tracks_out = []
    for tid, t in sorted(tracks_data.items(), key=lambda kv: kv[0]):
        if t["first_frame"] is None:
            continue
        first_f, last_f, seen = t["first_frame"], t["last_frame"], t["seen_frames"]
        max_risk = t.get("_max_risk", 0)
        events_list = list(t.get("_events_set", set()))
        tracks_out.append({
            "id": int(tid),
            "dwell_time_s": float(seen / float(fps)),
            "seen_frames": int(seen),
            "first_seen_s": float(first_f / float(fps)),
            "last_seen_s": float(last_f / float(fps)),
            "risk": int(max_risk),
            "state": "ALERT" if max_risk > 0 else "NORMAL",
            "events": events_list,
        })

    summary = {
        "created": _utc_now_iso(),
        "fps": float(fps),
        "frames": int(frame_count),
        "unique_people_estimate": int(len(tracks_out)),
        "total_dwell_time_s": float(sum(t["dwell_time_s"] for t in tracks_out)),
        "avg_dwell_time_s": float((sum(t["dwell_time_s"] for t in tracks_out) / len(tracks_out)) if tracks_out else 0.0),
    }

    return {
        "error": None,
        "output_video": out_basename,
        "heatmap": heatmap_basename,
        "tracks": tracks_out,
        "summary": summary,
    }
