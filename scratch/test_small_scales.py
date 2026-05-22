import cv2
import glob
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

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
bg = np.zeros((600, 800, 3), dtype=np.uint8)
bg[:] = (35, 28, 26)

for i, contour in enumerate(valid_contours[:3]):
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(sprite_sheet.shape[1], x + w + pad), min(sprite_sheet.shape[0], y + h + pad)
    crop = sprite_sheet[y1:y2, x1:x2]
    
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(crop_gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    print(f"\n=== {names[i]} ===")
    for scale in [0.15, 0.20, 0.25, 0.30, 0.40]:
        canvas = bg.copy()
        sh, sw = crop.shape[:2]
        tw = int(sw * scale)
        th = int(sh * scale)
        
        resized_crop = cv2.resize(crop, (tw, th))
        resized_mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        
        cy, cx = 300, 400
        u1, v1 = cx - tw // 2, cy - th // 2
        u2, v2 = u1 + tw, v1 + th
        
        bg_crop = canvas[v1:v2, u1:u2]
        alpha = resized_mask / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        blended = resized_crop * alpha + bg_crop * (1.0 - alpha)
        canvas[v1:v2, u1:u2] = blended.astype(np.uint8)
        
        results = model(canvas, conf=0.15, verbose=False)[0]
        print(f"  Scale {scale:.2f} (size {tw}x{th}):")
        if len(results.boxes) == 0:
            print("    NO DETECTIONS")
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"    Detected: {cls_name} ({conf:.2f})")
