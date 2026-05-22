import cv2
import glob
import numpy as np
from pathlib import Path

BRAIN_DIR = Path("/Users/a/.gemini/antigravity-ide/brain/84b4c0c8-bc55-4f33-93dc-bedffba1f6d5")
sprite_path = glob.glob(str(BRAIN_DIR / "mock_realistic_cars*.png"))[0]
img = cv2.imread(sprite_path)

# Print some pixels from the corners (background)
print("Top-left corner BGR values:")
print(img[0:5, 0:5])

# Print distribution of BGR values in the top-left 10x10 area
print("\nUnique values in top-left 10x10:")
print(np.unique(img[0:10, 0:10].reshape(-1, 3), axis=0))

# Print image shape
print(f"\nImage shape: {img.shape}")
