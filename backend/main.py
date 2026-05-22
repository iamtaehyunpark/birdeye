"""
main.py — BirdEye FastAPI application.

Endpoints
─────────
REST:
  GET  /api/status                 → system health + init state
  POST /api/upload/reference       → upload reference image
  GET  /api/reference-image        → serve reference image
  POST /api/init/auto              → auto-init H (SuperPoint→ORB chain)
  POST /api/init/manual            → compute H from clicked point pairs
  POST /api/reset                  → clear H + tracks

WebSocket:
  /ws/frames  ← binary JPEG frames from browser webcam
               → JSON result: {type, init_status, fps, detections,
                               static_detections, tracks, bg_ready}

Static:
  /  → serves frontend/index.html  (and all frontend assets)
"""
import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Set

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from background import BackgroundDifferencer
from detector import get_detector
from initializer import (
    compute_homography_superpoint_lightglue,
    compute_homography_manual,
)
from projector import project_point
from state import state
from tracker import MultiTracker

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("birdeye")

MODEL_PATH     = os.getenv("MODEL_PATH", "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))
HOST           = os.getenv("HOST", "0.0.0.0")
PORT           = int(os.getenv("PORT", "8000"))
MAX_AGE_SEC    = float(os.getenv("TRACKER_MAX_AGE_SEC", "2.0"))
MAX_DIST       = float(os.getenv("TRACKER_MAX_DIST", "150.0"))
MIN_HITS       = int(os.getenv("TRACKER_MIN_HITS", "2"))
BG_DIFF_THRESH = int(os.getenv("BG_DIFF_THRESHOLD", "40"))

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="BirdEye", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend at root — must be mounted AFTER all API routes are declared
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ── Global singletons ──────────────────────────────────────────────────────────
tracker   = MultiTracker(max_age_sec=MAX_AGE_SEC, max_dist=MAX_DIST, min_hits=MIN_HITS)
bg_diff   = BackgroundDifferencer(diff_threshold=BG_DIFF_THRESH)
_fps_buf: deque = deque(maxlen=60)   # rolling window of frame timestamps


def _fps() -> float:
    now = time.monotonic()
    _fps_buf.append(now)
    recent = [t for t in _fps_buf if now - t <= 2.0]
    return round(len(recent) / 2.0, 1) if len(recent) > 1 else 0.0


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    with state.lock:
        return {
            "init_status":    state.init_status,
            "init_error":     state.init_error,
            "init_info":      state.init_info,
            "has_reference":  state.ref_image_path is not None,
            "ref_image_size": state.ref_image_size,
            "has_homography": state.H is not None,
            "track_count":    len(tracker.tracks),
            "model_path":     MODEL_PATH,
        }


@app.post("/api/upload/reference")
async def upload_reference(file: UploadFile = File(...)):
    """Accept any flat image as the reference map."""
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image — check file format")

    out_path = UPLOAD_DIR / "reference.jpg"
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    h, w = img.shape[:2]
    with state.lock:
        state.ref_image_path = str(out_path)
        state.ref_image_size = (w, h)
        state.H              = None
        state.init_status    = "idle"
        state.init_error     = None
        state.init_info      = None

    tracker.reset()
    logger.info(f"Reference image uploaded: {w}×{h} → {out_path}")
    return {"status": "ok", "width": w, "height": h}


@app.get("/api/reference-image")
async def serve_reference():
    with state.lock:
        path = state.ref_image_path
    if not path or not Path(path).exists():
        raise HTTPException(404, "No reference image uploaded yet")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/init/mock-scenario")
async def init_mock_scenario():
    """Load the generated mock reference map and pre-computed homography."""
    mock_ref_path = UPLOAD_DIR / "mock_reference.jpg"
    if not mock_ref_path.exists():
        raise HTTPException(404, "Mock reference image not found. Run generate_mock_data.py first.")

    img = cv2.imread(str(mock_ref_path))
    if img is None:
        raise HTTPException(500, "Failed to read mock reference image")

    out_path = UPLOAD_DIR / "reference.jpg"
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    h, w = img.shape[:2]

    # Auto-load pre-computed homography if available
    mock_h_path = UPLOAD_DIR / "mock_homography.npy"
    H: Optional[np.ndarray] = None
    init_info: Optional[dict] = None
    if mock_h_path.exists():
        try:
            loaded = np.load(str(mock_h_path))
            if loaded.shape == (3, 3):
                H = loaded
                init_info = {"method": "mock_precomputed", "n_inliers": 4, "n_pairs": 4}
        except Exception as exc:
            logger.warning(f"Could not load mock homography: {exc}")

    with state.lock:
        state.ref_image_path = str(out_path)
        state.ref_image_size = (w, h)
        state.H              = H
        state.init_status    = "done" if H is not None else "idle"
        state.init_error     = None
        state.init_info      = init_info

    tracker.reset()
    bg_diff.invalidate()
    logger.info(f"Mock reference image loaded: {w}×{h}, homography={'yes' if H is not None else 'no'}")
    return {"status": "ok", "width": w, "height": h, "homography_loaded": H is not None}


@app.post("/api/init/mock-homography")
async def init_mock_homography():
    """Load the pre-computed mock homography matrix into state."""
    mock_h_path = UPLOAD_DIR / "mock_homography.npy"
    if not mock_h_path.exists():
        raise HTTPException(404, "mock_homography.npy not found. Run generate_mock_data.py first.")

    try:
        H = np.load(str(mock_h_path))
        if H.shape != (3, 3):
            raise ValueError(f"Expected 3×3, got {H.shape}")
    except Exception as exc:
        raise HTTPException(500, f"Failed to load homography: {exc}")

    info = {"method": "mock_precomputed", "n_inliers": 4, "n_pairs": 4}
    with state.lock:
        state.H           = H
        state.init_status = "done"
        state.init_error  = None
        state.init_info   = info

    tracker.reset()
    bg_diff.invalidate()
    logger.info("Mock homography loaded via /api/init/mock-homography")
    return {"status": "done", "info": info}


@app.get("/api/mock-video")
async def serve_mock_video():
    """Serve the generated mock camera feed video."""
    video_path = UPLOAD_DIR / "mock_camera.mp4"
    if not video_path.exists():
        video_path = UPLOAD_DIR / "mock_camera.avi"
    if not video_path.exists():
        raise HTTPException(404, "Mock camera video not found. Run generate_mock_data.py first.")
    
    media_type = "video/mp4" if video_path.suffix == ".mp4" else "video/avi"
    return FileResponse(str(video_path), media_type=media_type)



@app.post("/api/init/auto")
async def init_auto(body: dict):
    """
    Trigger automatic H computation from a camera snapshot.

    Body: { "frame_b64": "<base64-encoded JPEG>" }
    """
    with state.lock:
        if state.init_status == "running":
            return JSONResponse({"error": "Already running"}, status_code=409)
        if not state.ref_image_path:
            return JSONResponse({"error": "Upload a reference image first"}, status_code=400)
        state.init_status = "running"
        ref_path = state.ref_image_path

    # Decode camera snapshot
    try:
        raw   = base64.b64decode(body.get("frame_b64", ""))
        arr   = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode camera frame")
    except Exception as exc:
        with state.lock:
            state.init_status = "error"
            state.init_error  = str(exc)
        raise HTTPException(400, str(exc))

    ref_img = cv2.imread(ref_path)

    # Run in thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    H, info = await loop.run_in_executor(
        None,
        compute_homography_superpoint_lightglue,
        frame,
        ref_img,
    )

    with state.lock:
        if H is not None:
            state.H           = H
            state.init_status = "done"
            state.init_error  = None
            state.init_info   = info
        else:
            state.init_status = "error"
            state.init_error  = info.get("error", "Unknown")
            state.init_info   = info

    if H is not None:
        tracker.reset()
        bg_diff.invalidate()   # will re-setup lazily on next WS frame
        return {"status": "done", "info": info}
    else:
        return JSONResponse({"status": "error", "info": info}, status_code=422)


@app.post("/api/init/manual")
async def init_manual(body: dict):
    """
    Compute H from user-clicked point correspondences.

    Body: { "cam_pts": [[u,v],...], "ref_pts": [[x,y],...] }  (≥ 4 pairs)
    """
    cam_pts: list = body.get("cam_pts", [])
    ref_pts: list = body.get("ref_pts", [])

    if len(cam_pts) < 4 or len(cam_pts) != len(ref_pts):
        raise HTTPException(400, "Provide at least 4 matching pairs (cam_pts / ref_pts same length)")

    loop = asyncio.get_event_loop()
    H, info = await loop.run_in_executor(None, compute_homography_manual, cam_pts, ref_pts)

    with state.lock:
        if H is not None:
            state.H           = H
            state.init_status = "done"
            state.init_error  = None
            state.init_info   = info
        else:
            state.init_status = "error"
            state.init_error  = info.get("error", "Unknown")

    if H is not None:
        tracker.reset()
        bg_diff.invalidate()   # will re-setup lazily on next WS frame
        return {"status": "done", "info": info}
    else:
        return JSONResponse({"status": "error", "info": info}, status_code=422)


@app.post("/api/reset")
async def api_reset():
    with state.lock:
        state.H           = None
        state.init_status = "idle"
        state.init_error  = None
        state.init_info   = None
    tracker.reset()
    bg_diff.invalidate()
    _fps_buf.clear()
    return {"status": "reset"}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    """
    Bidirectional WebSocket for real-time processing.

    Receives: binary JPEG frames from the browser webcam.
    Sends:    JSON result packets at each frame.
    """
    await ws.accept()
    logger.info("WebSocket /ws/frames connected")

    detector = get_detector(MODEL_PATH, CONF_THRESHOLD)
    loop     = asyncio.get_event_loop()

    try:
        while True:
            msg = await ws.receive()

            # ── Disconnect ─────────────────────────────────────────────────
            if msg.get("type") == "websocket.disconnect":
                break

            # ── Keep-alive text ping ───────────────────────────────────────
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))
                except Exception:
                    pass
                continue

            frame_bytes = msg.get("bytes")
            if not frame_bytes:
                continue

            # ── Decode frame ───────────────────────────────────────────────
            arr   = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            fps = _fps()

            # ── Read shared state (non-blocking) ───────────────────────────
            with state.lock:
                H           = state.H.copy() if state.H is not None else None
                init_status = state.init_status

            if H is None:
                # Not initialized — report status only
                await ws.send_text(json.dumps({
                    "type":        "status",
                    "init_status": init_status,
                    "fps":         fps,
                    "detections":  [],
                    "tracks":      [],
                }))
                continue

            # ── Lazy BG differencer setup (first frame after H is ready) ───
            if not bg_diff.is_ready():
                with state.lock:
                    ref_path_now = state.ref_image_path
                if ref_path_now:
                    ref_bgr = cv2.imread(ref_path_now)
                    fh, fw  = frame.shape[:2]
                    bg_diff.setup(ref_bgr, H, fw, fh)

            # ── Detect ────────────────────────────────────────────────────────
            raw_dets = await loop.run_in_executor(None, detector.detect, frame)

            # Keep only vehicle classes to prevent false detections (e.g. sports ball, person, kite)
            # from polluting the tracking visualization.
            vehicle_classes = {"car", "truck", "bus", "motorcycle"}
            raw_dets = [d for d in raw_dets if d["class_name"] in vehicle_classes]

            # ── Background mask ───────────────────────────────────────────────
            # Pixels that differ from warped reference = potential moving objects
            bg_mask = bg_diff.compute_mask(frame)

            # ── Project ground-contact points to reference image ──────────────
            projected = []
            for det in raw_dets:
                u, v   = det["ground_px"]
                rx, ry = project_point(u, v, H)
                projected.append({
                    **det,
                    "ref_x":      rx,
                    "ref_y":      ry,
                    "class_name": det["class_name"],
                })

            # ── Filter: keep only detections absent from the reference ─────────
            # moving_dets → new objects in camera not present in reference image
            # static_dets → objects already in the reference (or background noise)
            if bg_diff.is_ready():
                moving_dets, static_dets = bg_diff.filter_detections(projected, bg_mask)
            else:
                moving_dets, static_dets = projected, []

            # ── Track only moving objects ──────────────────────────────────────
            track_list = tracker.update(moving_dets)

            # ── Serialize and broadcast ────────────────────────────────────────
            def _ser(d):
                return {
                    "class":      d["class_name"],
                    "confidence": round(d["confidence"], 3),
                    "bbox":       [round(x, 1) for x in d["bbox"]],
                    "ground_px":  [round(x, 1) for x in d["ground_px"]],
                    "ref_px":     [round(d["ref_x"], 1), round(d["ref_y"], 1)],
                }

            payload = {
                "type":               "result",
                "init_status":        init_status,
                "fps":                fps,
                "bg_ready":           bg_diff.is_ready(),
                "detections":         [_ser(d) for d in moving_dets],
                "static_detections":  [_ser(d) for d in static_dets],
                "tracks":             track_list,
            }
            await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/frames disconnected")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}", exc_info=True)


# ── Static frontend (must be mounted last, after all API routes) ───────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {"api": "/api/docs", "note": "frontend/ directory not found"}


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
