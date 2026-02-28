"""
Live webcam capture: stream from laptop camera, run YOLO, save output video and heatmap.
Press 'q' to stop. Output is written to smartshop_outputs and registered for the frontend.
"""
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone

from analytics_engine import get_model_path

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
    heatmap_grid = np.zeros((32, 32), dtype=np.float32)
    print("Live capture started. Press 'q' in the window to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)
        for box in results[0].boxes:
            xy = box.xyxy[0].cpu().numpy()
            cx = int((xy[0] + xy[2]) / 2 * 32 / w) % 32
            cy = int((xy[1] + xy[3]) / 2 * 32 / h) % 32
            heatmap_grid[cy, cx] += 1
        cv2.imshow("Smart Shop Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
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
    latest = {
        "output_video": out_basename,
        "heatmap": heatmap_basename,
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(LATEST_JSON, "w") as f:
        json.dump(latest, f, indent=2)
    print("Saved:", out_path, "| Heatmap:", heatmap_basename)


if __name__ == "__main__":
    main()
