import cv2
import glob
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolov8n.pt")

# Load sprite sheet
BRAIN_DIR = Path("/Users/a/.gemini/antigravity-ide/brain/84b4c0c8-bc55-4f33-93dc-bedffba1f6d5")
sprite_files = glob.glob(str(BRAIN_DIR / "mock_realistic_cars*.png"))
sprite_path = sprite_files[0]
sprite_sheet = cv2.imread(sprite_path)

gray = cv2.cvtColor(sprite_sheet, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

names = ["Red Sedan", "Blue SUV", "White Pickup"]

# Test backgrounds:
# Let's create a dark slate background of 800x600
bg = np.zeros((600, 800, 3), dtype=np.uint8)
bg[:] = (35, 28, 26)

for i, contour in enumerate(valid_contours[:3]):
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(sprite_sheet.shape[1], x + w + pad), min(sprite_sheet.shape[0], y + h + pad)
    
    crop = sprite_sheet[y1:y2, x1:x2]
    
    # 1. Threshold-based mask (current way)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask_thresh = cv2.threshold(crop_gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Contour-based solid mask (proposed way)
    mask_solid = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    shifted_contour = contour - [x1, y1]
    cv2.drawContours(mask_solid, [shifted_contour], -1, 255, -1)
    
    for mask_type, mask in [("threshold_mask", mask_thresh), ("solid_mask", mask_solid)]:
        # Let's test at scale 0.5 and 0.8
        for scale in [0.5, 0.8]:
            canvas = bg.copy()
            # Resize
            sh, sw = crop.shape[:2]
            tw = int(sw * scale)
            th = int(sh * scale)
            
            resized_crop = cv2.resize(crop, (tw, th))
            resized_mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            
            # Place in the middle of canvas
            cy, cx = 300, 400
            u1, v1 = cx - tw // 2, cy - th // 2
            u2, v2 = u1 + tw, v1 + th
            
            bg_crop = canvas[v1:v2, u1:u2]
            alpha = resized_mask / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            blended = resized_crop * alpha + bg_crop * (1.0 - alpha)
            canvas[v1:v2, u1:u2] = blended.astype(np.uint8)
            
            # Run YOLO
            results = model(canvas, conf=0.15, verbose=False)[0]
            print(f"Sprite {i} ({names[i]}), Mask: {mask_type}, Scale: {scale}")
            if len(results.boxes) == 0:
                print("  NO DETECTIONS")
            for box in results.boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"  Detected: {cls_name} ({conf:.2f})")
            print()
