# Smart Shop Analytics

Person detection, tracking (IDs, dwell time), age/gender estimation, heatmaps, and output video.  
**Best experience on Mac/Linux** (InsightFace for accurate age/gender). Windows uses a slower fallback unless you install C++ Build Tools.

---

## How to run (Mac / Linux recommended)

Use the project folder as the working directory.

### 1. Install dependencies

**Python (backend):**
```bash
pip install -r backend/requirements.txt
```
On Mac/Linux, this installs **InsightFace** + **onnxruntime** for accurate, fast age/gender. First run may download models.

**Node (frontend):**
```bash
npm install
```

### 2. Start the backend API

```bash
python backend/api_server.py
```
Leave it running. You should see: `Running on http://127.0.0.1:8000`.

### 3. Start the frontend

Open a **new** terminal, same project folder:

```bash
npm run dev
```
Then open **http://localhost:5173** in your browser.

You can:
- **Upload a video** — backend processes it; output video, heatmap, and tracking (IDs, age, gender, dwell time) appear on the dashboard.
- **Live camera** — click "Start camera" then "Record 10s"; the clip is uploaded and processed.
- Optionally run **`python backend/live_fast.py`** in a third terminal for desktop webcam capture (press **q** to stop).

---

**Summary:**  
Terminal 1: `python backend/api_server.py`  
Terminal 2: `npm run dev` → open http://localhost:5173  
Terminal 3 (optional): `python backend/live_fast.py` for webcam capture

---

## Windows / low-end machines

- **InsightFace** on Windows often needs **Microsoft Visual C++ Build Tools**. If `pip install insightface` fails with a C++ error, the app falls back to **DeepFace** (slower; age/gender runs less often).
- If processing is too slow, disable age/gender so only detection + tracking + heatmap run:
  ```powershell
  $env:DISABLE_AGE_GENDER="1"
  python backend/api_server.py
  ```

---

## Project structure

```text
  backend/
    api_server.py      # Flask API
    live_fast.py       # Webcam capture script
    analytics_engine.py # YOLO + tracking + age/gender + heatmap
    requirements.txt
    models/            # yolov8n.pt (copy or download)
    smartshop_outputs/
    uploads/
  src/
  index.html
  package.json
```

## Notes

- **YOLO model:** Use `backend/models/yolov8n.pt` (or place `yolov8n.pt` in project root; it will be copied once).
- **Age/Gender:** InsightFace (Mac/Linux) = best accuracy, runs every 5 frames. DeepFace fallback (e.g. Windows) = every 30 frames, max 2 people per frame.
- Outputs are written to `backend/smartshop_outputs/`.
