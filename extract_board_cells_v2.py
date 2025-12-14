import cv2
import numpy as np
from pathlib import Path

IMG = "IMG_2050.png"   # dark mode screenshot
OUTDIR = Path("debug")
OUTDIR.mkdir(exist_ok=True)

img = cv2.imread(IMG)
if img is None:
    raise SystemExit(f"Could not read image: {IMG}")

h, w = img.shape[:2]

# 1) Find board region by "colorfulness" (saturation) in HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

# Saturation threshold: board is colorful, background/UI mostly low saturation.
# (Tune S_MIN if needed, but this usually works on NYT Pips.)
S_MIN = 35
mask_color = (S > S_MIN).astype(np.uint8) * 255

# Clean up mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel, iterations=1)

cv2.imwrite(str(OUTDIR / "01_mask_color.png"), mask_color)

# Find largest connected component = board-ish region
cnts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise SystemExit("No contours found in color mask. Try lowering S_MIN.")

cnt = max(cnts, key=cv2.contourArea)
x, y, bw, bh = cv2.boundingRect(cnt)

# Add some padding to include dashed borders / rounded edges
pad = int(0.06 * max(bw, bh))
x0 = max(0, x - pad)
y0 = max(0, y - pad)
x1 = min(w, x + bw + pad)
y1 = min(h, y + bh + pad)

board = img[y0:y1, x0:x1].copy()
cv2.imwrite(str(OUTDIR / "02_board_roi.png"), board)

# 2) Within board ROI, detect cell blobs by "not background"
# Convert to LAB; background is dark/neutral, cells are lighter/colored.
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# Threshold on L to get lighter tiles; adaptive is more robust across screenshots
L_blur = cv2.GaussianBlur(L, (5, 5), 0)
mask_tiles = cv2.adaptiveThreshold(
    L_blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,  # block size
    -5   # C (negative pulls more foreground)
)

# Remove tiny specks / fill holes
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_tiles = cv2.morphologyEx(mask_tiles, cv2.MORPH_OPEN, k2, iterations=1)
mask_tiles = cv2.morphologyEx(mask_tiles, cv2.MORPH_CLOSE, k2, iterations=2)

cv2.imwrite(str(OUTDIR / "03_mask_tiles.png"), mask_tiles)

# 3) Connected components = candidate cells
num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_tiles, connectivity=8)

candidates = []
areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
if len(areas) == 0:
    raise SystemExit("No components found for tiles mask. Try adjusting adaptive params.")

# Estimate typical cell area using median of mid-sized components
areas_sorted = np.sort(areas)
median_area = np.median(areas_sorted[len(areas_sorted)//4 : 3*len(areas_sorted)//4])
if not np.isfinite(median_area) or median_area <= 0:
    median_area = np.median(areas_sorted)

min_area = int(median_area * 0.35)
max_area = int(median_area * 2.20)

dbg = board.copy()

for i in range(1, num):  # component id
    x, y, ww, hh, area = stats[i]
    if area < min_area or area > max_area:
        continue

    # Squareness / aspect filter (rounded squares)
    ar = ww / float(hh)
    if ar < 0.75 or ar > 1.33:
        continue

    # Also reject long skinny UI chunks
    if ww < 20 or hh < 20:
        continue

    candidates.append((x, y, ww, hh, area))

# Draw results
for x, y, ww, hh, area in candidates:
    cv2.rectangle(dbg, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

cv2.imwrite(str(OUTDIR / "04_cells_detected.png"), dbg)

print(f"Board ROI: x={x0}, y={y0}, w={x1-x0}, h={y1-y0}")
print(f"Estimated median cell area: {median_area:.1f}")
print(f"Detected {len(candidates)} cells → {OUTDIR / '04_cells_detected.png'}")

# Optionally dump cell boxes in original-image coordinates
cells_global = []
for x, y, ww, hh, area in candidates:
    cells_global.append((x0 + x, y0 + y, ww, hh))

# Save as simple text for your solver pipeline
with open("cells.txt", "w") as f:
    for x, y, ww, hh in sorted(cells_global, key=lambda t: (t[1], t[0])):
        f.write(f"{x},{y},{ww},{hh}\n")

print("Wrote cells.txt (x,y,w,h in full screenshot coords)")
