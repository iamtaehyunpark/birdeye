import cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")
video_path = "backend/uploads/mock_camera.mp4"
output_dir = Path("scratch/detection_frames")
output_dir.mkdir(exist_ok=True, parents=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 15 == 0:
        results = model(frame, conf=0.15, verbose=False)[0]
        annotated_frame = results.plot()
        out_path = output_dir / f"frame_{frame_idx:03d}.jpg"
        cv2.imwrite(str(out_path), annotated_frame)
        print(f"Saved annotated frame: {out_path}")
    frame_idx += 1

cap.release()
print("Done.")
