import cv2
import glob
import numpy as np
from pathlib import Path

BRAIN_DIR = Path("/Users/a/.gemini/antigravity-ide/brain/84b4c0c8-bc55-4f33-93dc-bedffba1f6d5")
sprite_path = glob.glob(str(BRAIN_DIR / "mock_realistic_cars*.png"))[0]
sprite_sheet = cv2.imread(sprite_path)
gray = cv2.cvtColor(sprite_sheet, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

names = ["Red Sedan", "Blue SUV", "White Pickup"]

for i, contour in enumerate(valid_contours[:3]):
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(sprite_sheet.shape[1], x + w + pad), min(sprite_sheet.shape[0], y + h + pad)
    
    crop = sprite_sheet[y1:y2, x1:x2]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask_thresh = cv2.threshold(crop_gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    mask_solid = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    shifted_contour = contour - [x1, y1]
    cv2.drawContours(mask_solid, [shifted_contour], -1, 255, -1)
    
    diff = cv2.absdiff(mask_thresh, mask_solid)
    diff_count = np.sum(diff > 0)
    print(f"{names[i]}: crop size {w+2*pad}x{h+2*pad}, differ at {diff_count} pixels ({(diff_count/mask_solid.size)*100:.1f}%)")
