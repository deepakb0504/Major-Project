"""
Live webcam capture: stream from laptop camera, run YOLO + risk (green->red),
save output video and heatmap. Same output style as upload (ID, risk, dwell, events).
Press 'q' to stop.
"""
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque

from analytics_engine import (
    get_model_path,
    CONFIDENCE_THRESHOLD,
    HEATMAP_GRID_SIZE,
    PERSON_CLASS_ID,
    ACCEL_THRESHOLD,
    HISTORY_LEN,
    RISK_DECAY_FRAMES,
    _risk_to_bgr,
    _draw_label,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "smartshop_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
LATEST_JSON = OUTPUT_DIR / "latest_live.json"


def main():
    try:
        model_path = get_model_path()
    except PermissionError as e:
        print(e)
        return
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print("YOLO load failed:", e)
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera (index 0). Check permissions or try another index.")
        return
    fps = 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_basename = f"output_live_{run_id}.mp4"
    heatmap_basename = f"heatmap_live_{run_id}.png"
    out_path = OUTPUT_DIR / out_basename
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    heatmap_grid = np.zeros((HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE), dtype=np.float32)
    tracks_data = defaultdict(lambda: {"first_frame": None, "last_frame": None, "seen_frames": 0})
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    track_risk = defaultdict(lambda: {"score": 0, "decay_counter": 0})
    frame_count = 0

    def decay_risks():
        for r in track_risk.values():
            if r["decay_counter"] > 0:
                r["decay_counter"] -= 1
            else:
                r["score"] = max(0, r["score"] - 1)

    print("Live capture started. Press 'q' in the window to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
                    v_prev_x, v_prev_y = (cx0 - cx1) / dt, (cy0 - cy1) / dt
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
                t["_max_risk"] = max(t.get("_max_risk", 0), risk)
                t["_events_set"] = t.get("_events_set", set()) | set(events)
                cx_i = int((x1 + x2) / 2 * HEATMAP_GRID_SIZE / max(w, 1)) % HEATMAP_GRID_SIZE
                cy_i = int((y1 + y2) / 2 * HEATMAP_GRID_SIZE / max(h, 1)) % HEATMAP_GRID_SIZE
                heatmap_grid[cy_i, cx_i] += 1.0
        decay_risks()
        writer.write(annotated)
        cv2.imshow("Smart Shop Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    if heatmap_grid.max() > 0:
        hm_norm = (heatmap_grid / heatmap_grid.max() * 255).astype(np.uint8)
        hm_resized = cv2.resize(hm_norm, (256, 256), interpolation=cv2.INTER_LINEAR)
        hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        cv2.imwrite(str(OUTPUT_DIR / heatmap_basename), hm_colored)
    else:
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(OUTPUT_DIR / heatmap_basename), blank)
    tracks_out = []
    for tid, t in sorted(tracks_data.items(), key=lambda kv: kv[0]):
        if t["first_frame"] is None:
            continue
        max_risk = t.get("_max_risk", 0)
        events_list = list(t.get("_events_set", set()))
        tracks_out.append({
            "id": int(tid),
            "dwell_time_s": float(t["seen_frames"] / float(fps)),
            "seen_frames": int(t["seen_frames"]),
            "first_seen_s": float(t["first_frame"] / float(fps)),
            "last_seen_s": float(t["last_frame"] / float(fps)),
            "risk": int(max_risk),
            "state": "ALERT" if max_risk > 0 else "NORMAL",
            "events": events_list,
        })
    summary = {
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fps": float(fps),
        "frames": int(frame_count),
        "unique_people_estimate": int(len(tracks_out)),
        "total_dwell_time_s": float(sum(t["dwell_time_s"] for t in tracks_out)),
        "avg_dwell_time_s": float((sum(t["dwell_time_s"] for t in tracks_out) / len(tracks_out)) if tracks_out else 0.0),
    }
    latest = {
        "output_video": out_basename,
        "heatmap": heatmap_basename,
        "created": summary["created"],
        "tracks": tracks_out,
        "summary": summary,
    }
    with open(LATEST_JSON, "w") as f:
        json.dump(latest, f, indent=2)
    print("Saved:", out_path, "| Heatmap:", heatmap_basename)


if __name__ == "__main__":
    main()
