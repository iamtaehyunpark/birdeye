import asyncio
import cv2
import json
import requests
import websockets
from pathlib import Path

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/frames"

async def test_pipeline():
    # 1. Load mock scenario
    print("Loading mock scenario...")
    res = requests.post(f"{API_URL}/api/init/mock-scenario")
    if res.status_code != 200:
        print("Failed to load mock scenario:", res.text)
        return
    print("Mock scenario loaded successfully.")

    # 2. Calibrate manually using ground truth targets
    print("Initializing homography calibration...")
    calib_data = {
        "cam_pts": [[250.0, 180.0], [1030.0, 150.0], [1120.0, 600.0], [180.0, 580.0]],
        "ref_pts": [[400.0, 300.0], [1400.0, 250.0], [1500.0, 850.0], [350.0, 800.0]]
    }
    res = requests.post(f"{API_URL}/api/init/manual", json=calib_data)
    if res.status_code != 200:
        print("Manual calibration failed:", res.text)
        return
    print("Homography ready.")

    # 3. Connect WebSocket and send video frames
    video_path = "backend/uploads/mock_camera.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    print("Connecting to WebSocket...")
    async with websockets.connect(WS_URL) as ws:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Send every 15 frames (1 second) to keep output readable
            if frame_idx % 15 == 0:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = jpeg.tobytes()
                
                await ws.send(jpeg_bytes)
                response = await ws.recv()
                data = json.loads(response)
                
                print(f"\n--- Frame {frame_idx} ---")
                print(f"  FPS: {data.get('fps')} | BG Ready: {data.get('bg_ready')}")
                
                moving = data.get("detections", [])
                static = data.get("static_detections", [])
                tracks = data.get("tracks", [])
                
                print("  Moving Detections:")
                if not moving:
                    print("    NONE")
                for m in moving:
                    print(f"    - {m['class']} ({m['confidence']:.2f}) at ground_px={m['ground_px']}, ref_px={m['ref_px']}")
                
                print("  Static Detections (Suppressed):")
                if not static:
                    print("    NONE")
                for s in static:
                    print(f"    - {s['class']} ({s['confidence']:.2f}) at ground_px={s['ground_px']}")
                
                print("  Active Tracks:")
                if not tracks:
                    print("    NONE")
                for t in tracks:
                    print(f"    - ID {t['id']}: {t['class']} at ({t['ref_x']:.1f}, {t['ref_y']:.1f}), hits={t.get('hits')}")
                    
            frame_idx += 1

    cap.release()

if __name__ == "__main__":
    asyncio.run(test_pipeline())
