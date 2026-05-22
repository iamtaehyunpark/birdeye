import cv2
import glob
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

output_dir = Path("scratch/sprites")
output_dir.mkdir(exist_ok=True, parents=True)

for i, contour in enumerate(valid_contours[:3]):
    x, y, w, h = cv2.boundingRect(contour)
    crop = sprite_sheet[y:y+h, x:x+w]
    cv2.imwrite(str(output_dir / f"sprite_{i}.png"), crop)
    print(f"Saved: {output_dir / f'sprite_{i}.png'}")
