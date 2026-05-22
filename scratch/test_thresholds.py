import cv2
import glob
import numpy as np
from pathlib import Path

BRAIN_DIR = Path("/Users/a/.gemini/antigravity-ide/brain/84b4c0c8-bc55-4f33-93dc-bedffba1f6d5")
sprite_path = glob.glob(str(BRAIN_DIR / "mock_realistic_cars*.png"))[0]
sprite_sheet = cv2.imread(sprite_path)
gray = cv2.cvtColor(sprite_sheet, cv2.COLOR_BGR2GRAY)

# Let's find the bounding box of the third sprite (White Pickup)
# We know from check_sprites: x=579, y=404, w=416, h=378
crop = sprite_sheet[404:404+378, 579:579+416]
crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

print("Transparency of White Pickup crop at different thresholds:")
for thresh_val in [240, 245, 250, 253, 254, 255]:
    # We want to keep everything that is NOT background
    # Since background is pure white (255, 255, 255), if we threshold at thresh_val:
    _, mask = cv2.threshold(crop_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    transparent_pct = (np.sum(mask == 0) / mask.size) * 100
    print(f"  Threshold {thresh_val}: {transparent_pct:.1f}% transparent pixels")
