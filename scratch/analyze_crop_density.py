import cv2
import glob
import numpy as np
from pathlib import Path

BRAIN_DIR = Path("/Users/a/.gemini/antigravity-ide/brain/84b4c0c8-bc55-4f33-93dc-bedffba1f6d5")
sprite_path = glob.glob(str(BRAIN_DIR / "mock_realistic_cars*.png"))[0]
img = cv2.imread(sprite_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

for i, contour in enumerate(valid_contours[:3]):
    x, y, w, h = cv2.boundingRect(contour)
    crop = img[y:y+h, x:x+w]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Let's count non-white pixels along the rows and columns
    # We define a pixel as non-white if BGR value is not close to (255, 255, 255)
    non_white = np.any(crop < 245, axis=2)
    
    # Sum along rows (height) and columns (width)
    row_sums = np.sum(non_white, axis=1)
    col_sums = np.sum(non_white, axis=0)
    
    # Find active range (where there are non-white pixels)
    active_rows = np.where(row_sums > 5)[0]
    active_cols = np.where(col_sums > 5)[0]
    
    if len(active_rows) > 0 and len(active_cols) > 0:
        y_min, y_max = active_rows[0], active_rows[-1]
        x_min, x_max = active_cols[0], active_cols[-1]
        new_w = x_max - x_min + 1
        new_h = y_max - y_min + 1
        print(f"Sprite {i}: bounding box {w}x{h}.")
        print(f"  Active non-white region: x=[{x_min}, {x_max}] (w={new_w}), y=[{y_min}, {y_max}] (h={new_h}), aspect_ratio={new_w/new_h:.2f}")
    else:
        print(f"Sprite {i}: no active non-white region found!")
