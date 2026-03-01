"""
Analytics engine: process video with YOLO, face (age/gender), produce output video and heatmap.
"""
import os
import shutil
import uuid
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

OUTPUT_DIR = Path(__file__).resolve().parent / "smartshop_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
MODEL_PATH_LOCAL = MODELS_DIR / "yolov8n.pt"
MODELS_DIR.mkdir(exist_ok=True)

PERSON_CLASS_ID = 0

# Lazy-loaded face analyzer for age/gender (InsightFace primary, DeepFace fallback)
_face_app = None
_face_backend = None


def _get_face_analyzer():
    global _face_app, _face_backend
    if _face_app is not None or _face_backend == "none":
        return _face_app, _face_backend
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CPUExecutionProvider"], name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        _face_backend = "insightface"
        return _face_app, _face_backend
    except Exception:
        pass
    try:
        import deepface  # noqa: F401
        _face_app = "deepface"
        _face_backend = "deepface"
        return _face_app, _face_backend
    except Exception:
        pass
    _face_backend = "none"
    return None, "none"


def _run_face_analysis(frame_bgr, person_bboxes=None):
    """
    Run age/gender on frame. Returns list of (bbox_xyxy, age_int, gender_str).
    - InsightFace: full-frame detection; person_bboxes ignored.
    - DeepFace: crops each person_bbox and analyzes; returns one result per crop.
    """
    app, backend = _get_face_analyzer()
    out = []
    if backend == "insightface" and app is not None:
        try:
            faces = app.get(frame_bgr)
            for f in faces:
                bbox = getattr(f, "bbox", None)
                if bbox is None or len(bbox) < 4:
                    continue
                age = int(getattr(f, "age", 0) or 0)
                g = getattr(f, "gender", None)
                if g is None:
                    g = getattr(f, "sex", "U")
                if isinstance(g, (int, float)):
                    gender_str = "M" if g == 0 else "F"
                else:
                    gender_str = str(g).upper()[:1] if g else "U"
                out.append((bbox, age, gender_str))
        except Exception:
            pass
        return out
    if backend == "deepface" and person_bboxes is not None:
        from deepface import DeepFace
        h, w = frame_bgr.shape[:2]
        for (x1, y1, x2, y2) in person_bboxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                result = DeepFace.analyze(crop, actions=["age", "gender"], enforce_detection=False, silent=True)
                if isinstance(result, list):
                    result = result[0] if result else {}
                age = int(result.get("age") or 0)
                g = result.get("dominant_gender", result.get("gender", ""))
                gender_str = str(g).upper()[:1] if g else "U"
                out.append(((x1, y1, x2, y2), age, gender_str))
            except Exception:
                continue
        return out
    return out


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iou(a, b):
    # a, b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _draw_label(img, text, x, y):
    cv2.rectangle(img, (x, y - 18), (x + 8 * len(text) + 10, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def get_model_path():
    """
    Use YOLO model from backend/models/ so OneDrive doesn't lock the file.
    If only project-root yolov8n.pt exists, copy it once to backend/models/.
    """
    # If a directory named *.pt exists (common when someone extracts a Torch checkpoint),
    # Ultralytics can't load it. Only treat real files as valid model paths.
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
    # If the project has a yolov8n.pt.zip (Torch checkpoints are ZIPs internally),
    # users sometimes rename it; use it as a last-resort source.
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
    Run person detection + lightweight tracking, write annotated output video and a simple heatmap.
    Returns dict with output_video, heatmap paths (relative to OUTPUT_DIR) plus tracks + summary.
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
    # Browser-compatible encoding: use ffmpeg via imageio if available (H.264).
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

    heatmap_grid = np.zeros((32, 32), dtype=np.float32)
    frame_count = 0

    # Simple IOU-based tracker (lightweight, no extra deps)
    next_track_id = 1
    tracks_active = {}  # id -> {bbox, seen_frames, missed, first_frame, last_frame}
    tracks_all = {}  # id -> same as above, never deleted (for analytics output)
    max_missed = max(10, int(fps * 1.5))
    iou_threshold = 0.25

    def _update_tracks(person_boxes, frame_idx):
        nonlocal next_track_id, tracks_active, tracks_all
        updated_ids = set()
        # Greedy matching by best IoU
        for pb in person_boxes:
            best_id = None
            best_iou = 0.0
            for tid, t in tracks_active.items():
                if tid in updated_ids:
                    continue
                i = _iou(pb, t["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_id = tid
            if best_id is not None and best_iou >= iou_threshold:
                t = tracks_active[best_id]
                t["bbox"] = pb
                t["seen_frames"] += 1
                t["missed"] = 0
                t["last_frame"] = frame_idx
                tracks_all[best_id] = dict(t)
                updated_ids.add(best_id)
            else:
                tid = next_track_id
                next_track_id += 1
                tnew = {
                    "bbox": pb,
                    "seen_frames": 1,
                    "missed": 0,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "age_samples": [],
                    "gender_samples": [],
                }
                tracks_active[tid] = tnew
                tracks_all[tid] = dict(tnew)
                updated_ids.add(tid)

        # Ensure age/gender lists exist for existing tracks
        for t in tracks_active.values():
            t.setdefault("age_samples", [])
            t.setdefault("gender_samples", [])

        # Age out tracks not updated
        to_del = []
        for tid, t in tracks_active.items():
            if tid not in updated_ids:
                t["missed"] += 1
                tracks_all[tid] = dict(t)
                if t["missed"] > max_missed:
                    to_del.append(tid)
        for tid in to_del:
            del tracks_active[tid]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if w == 0 or h == 0:
            h, w = frame.shape[:2]
            # If OpenCV writer was created with dummy size, re-create it now.
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
        results = model(frame, verbose=False)
        annotated = frame.copy()

        # Extract person detections
        person_boxes = []
        try:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                cls = boxes.cls.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else None
                for i in range(len(xyxy)):
                    if cls[i] != PERSON_CLASS_ID:
                        continue
                    if conf is not None and float(conf[i]) < 0.35:
                        continue
                    x1, y1, x2, y2 = xyxy[i]
                    x1 = int(max(0, min(w - 1, x1)))
                    y1 = int(max(0, min(h - 1, y1)))
                    x2 = int(max(0, min(w - 1, x2)))
                    y2 = int(max(0, min(h - 1, y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    person_boxes.append((x1, y1, x2, y2))
        except Exception:
            person_boxes = []

        _update_tracks(person_boxes, frame_count)

        # Age/gender: InsightFace (Mac/Linux, best accuracy) = every 5 frames, all faces.
        # DeepFace fallback (e.g. Windows without C++ build tools) = every 30 frames, max 2 people.
        _get_face_analyzer()  # ensure _face_backend is set
        run_face = (
            os.environ.get("DISABLE_AGE_GENDER") != "1"
            and tracks_active
            and (
                (_face_backend == "insightface" and frame_count % 5 == 0)
                or (_face_backend == "deepface" and frame_count % 30 == 0)
            )
        )
        if run_face:
            if _face_backend == "insightface":
                person_bboxes_for_face = None
            else:
                person_bboxes_for_face = sorted(
                    [t["bbox"] for t in tracks_active.values()],
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    reverse=True,
                )[:2]
            face_results = _run_face_analysis(frame, person_bboxes=person_bboxes_for_face)
            for fbbox, age_val, gender_str in face_results:
                fbbox = (float(fbbox[0]), float(fbbox[1]), float(fbbox[2]), float(fbbox[3]))
                best_tid = None
                best_iou_val = 0.0
                for tid, t in tracks_active.items():
                    i = _iou(t["bbox"], fbbox)
                    if i > best_iou_val:
                        best_iou_val = i
                        best_tid = tid
                if best_tid is not None and best_iou_val >= 0.15:
                    t = tracks_active[best_tid]
                    t.setdefault("age_samples", []).append(age_val)
                    t.setdefault("gender_samples", []).append(gender_str)
                    tracks_all[best_tid] = dict(t)

        # Draw + heatmap using track boxes (more stable than raw detections)
        for tid, t in list(tracks_active.items()):
            x1, y1, x2, y2 = t["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            dwell_s = t["seen_frames"] / float(fps)
            age_samples = t.get("age_samples") or []
            gender_samples = t.get("gender_samples") or []
            age_str = str(int(np.median(age_samples))) if age_samples else ""
            gender_str = Counter(gender_samples).most_common(1)[0][0] if gender_samples else ""
            label = f"ID {tid}  {dwell_s:.1f}s"
            if age_str or gender_str:
                label += f"  {gender_str}{age_str}"
            _draw_label(annotated, label, x1, max(20, y1))
            cx = int((x1 + x2) / 2 * 32 / max(w, 1)) % 32
            cy = int((y1 + y2) / 2 * 32 / max(h, 1)) % 32
            heatmap_grid[cy, cx] += 1.0

        if writer_kind == "imageio":
            # imageio expects RGB
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

    # Save heatmap
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

    # Build tracking analytics (median age, mode gender for stability)
    tracks_out = []
    for tid, t in sorted(tracks_all.items(), key=lambda kv: kv[0]):
        age_samples = t.get("age_samples") or []
        gender_samples = t.get("gender_samples") or []
        age_est = int(round(np.median(age_samples))) if age_samples else None
        gender_est = Counter(gender_samples).most_common(1)[0][0] if gender_samples else None
        if gender_est == "U":
            gender_est = None
        tracks_out.append({
            "id": int(tid),
            "dwell_time_s": float(t["seen_frames"] / float(fps)),
            "seen_frames": int(t["seen_frames"]),
            "first_seen_s": float(t["first_frame"] / float(fps)),
            "last_seen_s": float(t["last_frame"] / float(fps)),
            "age": age_est,
            "gender": gender_est,
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
