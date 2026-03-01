"""
Real-time live risk analysis: per-frame YOLO tracking + velocity/acceleration
for risk score, state (NORMAL/ALERT), and events (e.g. SUDDEN_ACCEL).
"""
import time
import numpy as np
from collections import defaultdict, deque

from analytics_engine import get_model_path, PERSON_CLASS_ID, CONFIDENCE_THRESHOLD

# Risk: sudden movement (acceleration) threshold in pixels per frame^2 (tuned for typical webcam res)
ACCEL_THRESHOLD = 15.0
# Max history length per track for velocity/acceleration
HISTORY_LEN = 8
# Risk score 0-5; SUDDEN_ACCEL adds 1, decays over time
RISK_DECAY_FRAMES = 30

# Same scale as analytics_engine: green (safe) -> red (theft)
RISK_HEX = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444", "#b91c1c"]


class LiveRiskEngine:
    def __init__(self):
        self._model = None
        self._track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self._track_risk = defaultdict(lambda: {"score": 0, "decay_counter": 0})
        self._frame_index = 0

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO
            path = get_model_path()
            self._model = YOLO(path)
        return self._model

    def reset(self):
        """Reset tracker state (e.g. new session)."""
        self._track_history.clear()
        self._track_risk.clear()
        self._frame_index = 0

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Run YOLO track on one frame, compute risk per track.
        frame_bgr: BGR numpy array (H, W, 3).
        Returns list of { id, bbox_xyxy, state, risk, events, dwell_s }.
        """
        self._frame_index += 1
        model = self._get_model()
        results = model.track(
            frame_bgr,
            persist=True,
            verbose=False,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            iou=0.45,
        )
        out_tracks = []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            self._decay_risks()
            return out_tracks

        h, w = frame_bgr.shape[:2]
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for i in range(len(ids)):
            tid = int(ids[i])
            x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            history = self._track_history[tid]
            risk_state = self._track_risk[tid]
            events = []

            # Velocity & acceleration from history
            if len(history) >= 2:
                (t0, cx0, cy0) = history[-1]
                (t1, cx1, cy1) = history[-2]
                dt = max(0.001, (t0 - t1))
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
            history.append((self._frame_index, cx, cy))

            risk = risk_state["score"]
            state = "NORMAL" if risk == 0 else "ALERT"
            dwell_s = len(history) / 30.0  # approximate; real fps not known here
            color_hex = RISK_HEX[min(5, risk)]

            out_tracks.append({
                "id": tid,
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "state": state,
                "risk": risk,
                "events": events,
                "dwell_s": round(dwell_s, 1),
                "color_hex": color_hex,
            })

        self._decay_risks()
        return out_tracks

    def _decay_risks(self):
        for tid, risk_state in list(self._track_risk.items()):
            if risk_state["decay_counter"] > 0:
                risk_state["decay_counter"] -= 1
            else:
                risk_state["score"] = max(0, risk_state["score"] - 1)


# Singleton for API (persistent tracker state across requests)
_engine = LiveRiskEngine()


def process_live_frame(frame_bgr: np.ndarray):
    return _engine.process_frame(frame_bgr)


def reset_live_session():
    _engine.reset()
