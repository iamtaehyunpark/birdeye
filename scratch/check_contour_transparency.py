import cv2
import glob
import numpy as np
from pathlib import Path

BRAIN_DIR = Path("/Users/a/.gemini/antigravity-ide/brain/84b4c0c8-bc55-4f33-93dc-bedffba1f6d5")
sprite_path = glob.glob(str(BRAIN_DIR / "mock_realistic_cars*.png"))[0]
sprite_sheet = cv2.imread(sprite_path)
gray = cv2.cvtColor(sprite_sheet, cv2.COLOR_BGR2GRAY)

# Threshold at 250 to segment from background
_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

x, y, w, h = cv2.boundingRect(valid_contours[2])
print(f"White Pickup bounding box: x={x}, y={y}, w={w}, h={h}")

# Create solid mask
mask_solid = np.zeros((h, w), dtype=np.uint8)
shifted_contour = valid_contours[2] - [x, y]
cv2.drawContours(mask_solid, [shifted_contour], -1, 255, -1)

solid_transparent_pct = (np.sum(mask_solid == 0) / mask_solid.size) * 100
print(f"Solid mask transparent percentage: {solid_transparent_pct:.1f}%")

# Let's compare with threshold mask at 250 on the crop:
crop = sprite_sheet[y:y+h, x:x+w]
crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
_, mask_thresh = cv2.threshold(crop_gray, 250, 255, cv2.THRESH_BINARY_INV)
thresh_transparent_pct = (np.sum(mask_thresh == 0) / mask_thresh.size) * 100
print(f"Threshold mask (250) transparent percentage: {thresh_transparent_pct:.1f}%")

# Difference: pixels that are inside the contour but masked out by threshold 250 (i.e. holes in the truck)
holes_mask = (mask_solid > 0) & (mask_thresh == 0)
holes_pct = (np.sum(holes_mask) / mask_solid.size) * 100
print(f"Holes inside the solid contour: {holes_pct:.1f}% of bounding box")
