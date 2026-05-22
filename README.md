# BirdEye — Edge-Cloud Geospatial Vision System

Real-time object detection + tracking projected onto **any flat reference image** — satellite, floorplan, blueprint, or any photograph. No map tiles, no cloud API, fully self-contained.

```
Camera Feed  →  YOLO Detection  →  H-Matrix Projection  →  Reference Image Overlay
```

---

## How It Works

| Phase | Component | When |
|-------|-----------|------|
| **Init** (once) | SuperPoint + LightGlue → RANSAC → **H matrix** | Camera boot / on demand |
| **Fallback init** | ORB + BFMatcher → RANSAC → H matrix | If LightGlue not installed |
| **Manual init** | User clicks 4+ point pairs → H matrix | Always available as backup |
| **Runtime** (30+ FPS) | YOLO → bottom-center ground point → `H @ [u,v,1]` → ref coords | Continuous |
| **Tracking** | Kalman Filter (constant-velocity) + Hungarian assignment | Continuous |
| **Display** | Canvas overlay: trail, velocity arrow, covariance circle | WebSocket push |

---

## Quick Start

```bash
# 1. Clone / open the repo
cd birdeye

# 2. (Optional but recommended) Install LightGlue for best auto-init accuracy
#    pip install git+https://github.com/cvg/LightGlue.git

# 3. Run
./run.sh
```

Then open **http://localhost:8000** in your browser.

---

## Configuration (`.env`)

Copy `.env.example` → `.env` and edit:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `yolov8n.pt` | Path to YOLO `.pt` weights |
| `CONF_THRESHOLD` | `0.35` | Detection confidence cutoff |
| `PORT` | `8000` | Server port |
| `TRACKER_MAX_AGE_SEC` | `2.0` | Seconds before a track expires |
| `TRACKER_MAX_DIST` | `150.0` | Max ref-px distance for track matching |
| `TRACKER_MIN_HITS` | `2` | Min detections before a track appears |

---

## Using Your Own YOLO Model

Train your model separately (in this repo or any other), then:

```bash
# In .env:
MODEL_PATH=/absolute/path/to/your_model.pt
```

The model is loaded once at startup. Any ultralytics-compatible `.pt` file works.

---

## Usage Walkthrough

1. **Open the dashboard** → `http://localhost:8000`
2. **Upload Reference Image** — click the upload zone in Step 01, drag & drop any flat image
3. **Connect Camera** — select your webcam from the dropdown, click Connect
4. **Calibrate** — choose one of:
   - **Auto-Initialize**: one click, SuperPoint+LightGlue computes H automatically
   - **Manual (4-pt)**: click 4+ matching points on camera + reference — guaranteed to work
5. **Watch** — detected objects appear as moving icons on your reference image in real-time

---

## Project Structure

```
birdeye/
├── backend/
│   ├── main.py          # FastAPI app + WebSocket pipeline
│   ├── initializer.py   # SuperPoint/ORB/Manual → H matrix
│   ├── detector.py      # YOLO inference wrapper
│   ├── tracker.py       # Kalman filter multi-object tracker
│   ├── projector.py     # H-matrix pixel → reference coord
│   ├── state.py         # Shared system state singleton
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── app.js           # Main orchestrator
│       ├── ws.js            # WebSocket client
│       ├── camera.js        # Webcam capture + frame sender
│       ├── renderer.js      # Canvas rendering (camera + reference)
│       └── calibration.js   # Manual 4-point calibration UI
├── .env.example
└── run.sh
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/status` | System health + init state |
| `POST` | `/api/upload/reference` | Upload reference image |
| `GET`  | `/api/reference-image` | Serve the reference image |
| `POST` | `/api/init/auto` | Auto-initialize H (body: `{frame_b64}`) |
| `POST` | `/api/init/manual` | Manual H from points (body: `{cam_pts, ref_pts}`) |
| `POST` | `/api/reset` | Reset H + tracks |
| `WS`   | `/ws/frames` | Stream frames → receive detection+track JSON |

Full interactive docs: **http://localhost:8000/api/docs**

---

## Notes on Cross-View Matching

- **Best case**: reference and camera images share visible structural features (lines, corners, patterns)
- **Works well**: indoor floorplan vs. surveillance camera, aerial vs. ground (partial overlap)
- **Hard case**: completely textureless scenes — use Manual 4-pt calibration
- SuperPoint+LightGlue is loaded, computes H, then **flushed from VRAM** — runtime uses zero DL compute
