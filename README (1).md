# Smart Shop Analytics

## How to run

Do these in order. Use the project folder as the working directory.

### 1. Install dependencies

**Python (backend):**
```bash
pip install -r backend/requirements.txt
```

**Node (frontend):**
```bash
npm install
```

### 2. Start the backend API

```bash
python backend/api_server.py
```

Leave this running. You should see something like: `Running on http://127.0.0.1:8000`.

### 3. Start the frontend

Open a **new terminal**, same project folder:

```bash
npm run dev
```

Then open in your browser: **http://localhost:5173**

You can now:
- **Upload a video** (button on the page) — the backend will process it and the **output video** and **heatmap** will appear on the dashboard.
- **Use live camera** — see step 4.

### 4. (Optional) Live camera capture

Open **another** terminal, same project folder:

```bash
python backend/live_fast.py
```

- A window will open with the laptop webcam and YOLO overlay.
- Press **q** in that window to stop. The last run is saved and will show as the latest output in the browser (refresh or wait a few seconds).

---

**Summary:**  
Terminal 1: `python backend/api_server.py`  
Terminal 2: `npm run dev` → open http://localhost:5173  
Terminal 3 (optional): `python backend/live_fast.py` for webcam capture

## Project structure

```text
Smart-shop-main/
  backend/
    app.py
    api_server.py
    live_fast.py
    analytics_engine.py
    requirements.txt
    models/
      yolov8n-cls.pt
    smartshop_outputs/
  public/
  src/
  index.html
  package.json
```

## Frontend (React + Vite)

```bash
npm install
npm run dev
```

Frontend expects analytics API at `http://127.0.0.1:8000` by default.
Override using:

```bash
set VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Backend (Python)

API service for local analytics processing + frontend downloads:

```bash
python backend/api_server.py
```

Live webcam script:

```bash
python backend/live_fast.py
```

Suggested Python dependencies:

```bash
pip install -r backend/requirements.txt
```

## Notes

- Classification model is expected at `backend/models/yolov8n-cls.pt`.
- Detection model uses `backend/models/yolov8n.pt` if present, otherwise `yolov8n.pt` in project root.
- Age/Gender uses InsightFace first, and falls back to DeepFace when InsightFace is unavailable.
- Generated outputs are written to `backend/smartshop_outputs/`.
- Uploading via frontend starts background analysis on local backend automatically.
- Frontend reads latest analytics from `GET /api/analytics/latest`.
