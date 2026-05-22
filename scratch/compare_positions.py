import numpy as np
import cv2
import math
from pathlib import Path
from ultralytics import YOLO

# Load homography and video
UPLOAD_DIR = Path("backend/uploads")
H_gt = np.load(str(UPLOAD_DIR / "mock_homography.npy"))
H_inv = np.linalg.inv(H_gt)

cam_w, cam_h = 1280, 720
target_a = (400, 300)
target_b = (1400, 250)
target_c = (1500, 850)
target_d = (350, 800)

targets_ref = [target_a, target_b, target_c, target_d]
targets_name = ['A', 'B', 'C', 'D']
targets_cam = []
for t in targets_ref:
    pt_ref = np.array([[[t[0], t[1]]]], dtype=np.float32)
    pt_cam = cv2.perspectiveTransform(pt_ref, H_inv)
    targets_cam.append(pt_cam[0][0])

total_frames = 300

def get_red_sedan_pos(frame_idx):
    start_x, start_y = 150.0, 520.0
    end_x = 1750.0
    t = frame_idx / total_frames
    x = start_x + (end_x - start_x) * t
    y = start_y + 10 * math.sin(t * 4 * math.pi)
    return x, y

def get_blue_suv_pos(frame_idx):
    start_x, start_y = 1600.0, 350.0
    end_x, end_y = 300.0, 850.0
    t = frame_idx / total_frames
    x = start_x + (end_x - start_x) * t
    y = start_y + (end_y - start_y) * t
    return x, y

def get_white_pickup_pos(frame_idx):
    if frame_idx < 30:
        return -500.0, -500.0
    start_x, start_y = 1700.0, 560.0
    end_x, end_y = 300.0, 560.0
    t = (frame_idx - 30) / (total_frames - 30)
    x = start_x + (end_x - start_x) * t
    y = start_y
    return x, y

# Load YOLO
model = YOLO("yolov8n.pt")
video_path = "backend/uploads/mock_camera.mp4"
cap = cv2.VideoCapture(video_path)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 30 == 0:
        print(f"\n--- Frame {frame_idx} ---")
        # Targets cam positions
        print("Expected Target Positions (Cam Space):")
        for name, pt in zip(targets_name, targets_cam):
            print(f"  Target {name}: [{pt[0]:.1f}, {pt[1]:.1f}]")
        
        # Vehicles cam positions
        print("Expected Vehicle Positions (Cam Space):")
        for vname, func in [('Red Sedan', get_red_sedan_pos), ('Blue SUV', get_blue_suv_pos), ('White Pickup', get_white_pickup_pos)]:
            rx, ry = func(frame_idx)
            if rx > 0:
                pt_ref = np.array([[[rx, ry]]], dtype=np.float32)
                pt_cam = cv2.perspectiveTransform(pt_ref, H_inv)
                print(f"  {vname}: [{pt_cam[0][0][0]:.1f}, {pt_cam[0][0][1]:.1f}]")
        
        # YOLO detections
        results = model(frame, conf=0.15, verbose=False)[0]
        print("YOLO Detections:")
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            bbox = [round(x, 1) for x in box.xyxy[0].tolist()]
            bc = [(bbox[0]+bbox[2])/2, bbox[3]]
            print(f"  {cls_name} ({conf:.2f}) at bbox {bbox}, bottom-center {bc}")

    frame_idx += 1
cap.release()
