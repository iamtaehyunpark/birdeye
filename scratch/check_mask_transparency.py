import cv2
import glob
import numpy as np
from pathlib import Path

BRAIN_DIR = Path("/Users/a/.gemini/antigravity-ide/brain/84b4c0c8-bc55-4f33-93dc-bedffba1f6d5")
sprite_files = glob.glob(str(BRAIN_DIR / "mock_realistic_cars*.png"))
sprite_path = sprite_files[0]
sprite_sheet = cv2.imread(sprite_path)
gray = cv2.cvtColor(sprite_sheet, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

for i, contour in enumerate(valid_contours[:3]):
    x, y, w, h = cv2.boundingRect(contour)
    crop = sprite_sheet[y:y+h, x:x+w]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, crop_mask = cv2.threshold(crop_gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    # Let's count how many pixels within the contour are masked out
    # Save the masked sprite on a black background
    masked = crop.copy()
    masked[crop_mask == 0] = [0, 0, 255] # Mark transparent as RED
    cv2.imwrite(f"scratch/sprites/masked_red_{i}.png", masked)
    
    # Let's also see what percentage of the bounding box is transparent
    total_pixels = w * h
    transparent_pixels = np.sum(crop_mask == 0)
    percent = (transparent_pixels / total_pixels) * 100
    print(f"Sprite {i}: size={w}x{h}, transparent={percent:.1f}%")
