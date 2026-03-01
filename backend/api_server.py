"""
Flask API: upload video, run analytics, serve output video and heatmap; live real-time risk API.
"""
import base64
import os
import uuid
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from analytics_engine import process_video, OUTPUT_DIR
from live_risk_engine import process_live_frame, reset_live_session

LATEST_LIVE_JSON = OUTPUT_DIR / "latest_live.json"

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

jobs = {}  # job_id -> { status, output_video?, heatmap?, error? }


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_job(job_id, input_path):
    result = process_video(input_path, job_id)
    jobs[job_id]["status"] = "completed" if not result.get("error") else "failed"
    jobs[job_id].update(result)
    try:
        os.remove(input_path)
    except OSError:
        pass


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400
    job_id = str(uuid.uuid4()).replace("-", "")[:12]
    ext = f.filename.rsplit(".", 1)[1].lower()
    save_name = secure_filename(f"{job_id}.{ext}")
    input_path = UPLOAD_DIR / save_name
    f.save(str(input_path))
    jobs[job_id] = {"status": "processing", "created": utc_now_iso()}
    threading.Thread(target=run_job, args=(job_id, str(input_path)), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/api/jobs/<job_id>", methods=["GET"])
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/api/analytics/latest", methods=["GET"])
def analytics_latest():
    base = request.host_url.rstrip("/")
    # Prefer live capture output if present
    if LATEST_LIVE_JSON.exists():
        try:
            with open(LATEST_LIVE_JSON) as f:
                live = json.load(f)
            ov, hm = live.get("output_video"), live.get("heatmap")
            if ov and (OUTPUT_DIR / ov).exists():
                return jsonify({
                    "output_video": f"{base}/api/files/{ov}",
                    "heatmap": f"{base}/api/files/{hm}" if hm and (OUTPUT_DIR / hm).exists() else None,
                    "tracks": live.get("tracks"),
                    "summary": live.get("summary"),
                    "output_video_basename": ov,
                    "heatmap_basename": hm,
                })
        except Exception:
            pass
    completed = [j for j in jobs.values() if j.get("status") == "completed"]
    if not completed:
        return jsonify({
            "output_video": None,
            "heatmap": None,
            "tracks": None,
            "summary": None,
            "output_video_basename": None,
            "heatmap_basename": None,
        })
    latest = max(completed, key=lambda x: x.get("created", ""))
    return jsonify({
        "output_video": f"{base}/api/files/{latest['output_video']}" if latest.get("output_video") else None,
        "heatmap": f"{base}/api/files/{latest['heatmap']}" if latest.get("heatmap") else None,
        "tracks": latest.get("tracks"),
        "summary": latest.get("summary"),
        "output_video_basename": latest.get("output_video"),
        "heatmap_basename": latest.get("heatmap"),
    })


@app.route("/api/files/<filename>", methods=["GET"])
def serve_file(filename):
    path = OUTPUT_DIR / filename
    if not path.is_file():
        return "Not found", 404
    ext = path.suffix.lower()
    mimetype = None
    if ext == ".mp4":
        mimetype = "video/mp4"
    elif ext == ".webm":
        mimetype = "video/webm"
    elif ext == ".png":
        mimetype = "image/png"
    elif ext == ".json":
        mimetype = "application/json"
    return send_file(path, as_attachment=False, download_name=filename, mimetype=mimetype, conditional=True)


@app.route("/api/live/reset", methods=["POST"])
def live_reset():
    """Reset live tracker state (e.g. when starting a new live session)."""
    try:
        reset_live_session()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/frame", methods=["POST"])
def live_frame():
    """
    Accept one frame (JSON { "image": "<base64>" } or multipart "frame" file).
    Returns { "tracks": [ { id, bbox_xyxy, state, risk, events, dwell_s } ] }.
    """
    frame_bgr = None
    if request.content_type and "application/json" in request.content_type:
        data = request.get_json(silent=True) or {}
        b64 = data.get("image") or data.get("base64")
        if not b64:
            return jsonify({"error": "Missing 'image' or 'base64' in JSON body"}), 400
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64: {e}"}), 400
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif "frame" in request.files:
        f = request.files["frame"]
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"error": "Could not decode image. Send JSON { image: base64 } or multipart 'frame' file."}), 400
    try:
        tracks = process_live_frame(frame_bgr)
        return jsonify({"tracks": tracks})
    except PermissionError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
