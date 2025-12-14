import cv2
import numpy as np
from pathlib import Path

IMG = "IMG_2050.png"
OUTDIR = Path("debug")
OUTDIR.mkdir(exist_ok=True)

img = cv2.imread(IMG)
if img is None:
    raise SystemExit(f"Could not read image: {IMG}")
H, W = img.shape[:2]

def save(p, im):
    cv2.imwrite(str(p), im)

# -----------------------------
# 1) Board ROI via saturation
# -----------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
S = hsv[:, :, 1]

S_MIN = 15  # <-- lowered from 25
mask_color = ((S > S_MIN) * 255).astype(np.uint8)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, k, iterations=2)
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN,  k, iterations=1)
save(OUTDIR / "v41_01_mask_color.png", mask_color)

cnts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise SystemExit("No ROI found. Lower S_MIN more (e.g. 8–12).")

cnt = max(cnts, key=cv2.contourArea)
x, y, bw, bh = cv2.boundingRect(cnt)

pad = int(0.07 * max(bw, bh))
x0 = max(0, x - pad)
y0 = max(0, y - pad)
x1 = min(W, x + bw + pad)
y1 = min(H, y + bh + pad)

board_full = img[y0:y1, x0:x1].copy()
save(OUTDIR / "v41_02_board_roi.png", board_full)

CROP_BOTTOM = 0.74  # <-- more aggressive; tray should be gone
bh2, bw2 = board_full.shape[:2]
board = board_full[: int(bh2 * CROP_BOTTOM), :].copy()
save(OUTDIR / "v41_02b_board_roi_cropped.png", board)

# -----------------------------
# Helpers
# -----------------------------
def cleanup(mask, open_k=0, close_k=9, open_it=1, close_it=2):
    m = mask.copy()
    if open_k > 0:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1, iterations=open_it)
    if close_k > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=close_it)
    return m

def boxes_from_components(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return []

    boxes = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Loosened a lot:
        if w < 14 or h < 14:
            continue
        ar = w / float(h)
        if ar < 0.55 or ar > 1.80:
            continue
        if area < 120:
            continue

        boxes.append((x, y, w, h, area))
    return boxes

def score_boxes(boxes):
    # No hard fail. Just prefer more + consistent sizing.
    n = len(boxes)
    if n == 0:
        return -1e9

    areas = np.array([b[4] for b in boxes], dtype=np.float32)
    sizes = np.array([(b[2] + b[3]) / 2.0 for b in boxes], dtype=np.float32)
    area_cv = float(areas.std() / (areas.mean() + 1e-6))
    size_cv = float(sizes.std() / (sizes.mean() + 1e-6))

    # prefer roughly 12–24, but don't nuke other counts
    target = 16.0
    count_penalty = abs(n - target) / target

    return (n / (1.0 + 2.5*area_cv + 2.0*size_cv)) * (1.0 / (1.0 + 0.7*count_penalty))

def normalize_foreground(mask):
    # Choose invert/non-invert based on fill ratio
    fill = mask.mean() / 255.0
    # If it's mostly white, invert; if mostly black, keep; else keep.
    if fill > 0.70:
        return cv2.bitwise_not(mask), "inv_auto"
    return mask, "as_is"

# -----------------------------
# 2) Candidate generation
# -----------------------------
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
L = lab[:, :, 0]

cands = []

# Otsu
for blur in [3, 5, 7]:
    Lb = cv2.GaussianBlur(L, (blur, blur), 0)
    _, m = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m, tag = normalize_foreground(m)
    cands.append((f"otsu_b{blur}_{tag}", m))

# Adaptive
for block in [21, 31, 41, 51]:
    for C in [-3, -7, -11]:
        Lb = cv2.GaussianBlur(L, (5, 5), 0)
        m = cv2.adaptiveThreshold(Lb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block, C)
        m, tag = normalize_foreground(m)
        cands.append((f"adapt_b{block}_C{C}_{tag}", m))

morph_grid = [(0, 5), (0, 7), (0, 9), (3, 7), (5, 9), (7, 11)]

best = None
top5 = []  # keep best masks for inspection

for name, base in cands:
    for ok, ck in morph_grid:
        m = cleanup(base, open_k=ok, close_k=ck)
        boxes = boxes_from_components(m)

        sc = score_boxes(boxes)
        record = (sc, name, ok, ck, m, boxes)

        top5.append(record)
        top5 = sorted(top5, key=lambda t: t[0], reverse=True)[:5]

        if best is None or sc > best["score"]:
            best = {"name": name, "ok": ok, "ck": ck, "mask": m, "boxes": boxes, "score": sc}

# Save top 5 masks + overlays
for rank, (sc, name, ok, ck, m, boxes) in enumerate(top5, start=1):
    save(OUTDIR / f"v41_top{rank:02d}_mask_{sc:.3f}_{name}_o{ok}_c{ck}.png", m)
    overlay = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    vis = board.copy()
    for x, y, w, h, _ in boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    save(OUTDIR / f"v41_top{rank:02d}_boxes_{sc:.3f}_{name}_o{ok}_c{ck}.png", vis)

# -----------------------------
# 3) Output best
# -----------------------------
dbg = board.copy()
for x, y, w, h, _ in best["boxes"]:
    cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
save(OUTDIR / "v41_best_boxes.png", dbg)
save(OUTDIR / "v41_best_mask.png", best["mask"])

print(f"ROI: x={x0}, y={y0}, w={x1-x0}, h={y1-y0} | crop_bottom={CROP_BOTTOM}")
print(f"BEST: {best['name']} | open={best['ok']} close={best['ck']} | score={best['score']:.3f}")
print(f"Detected {len(best['boxes'])} boxes")
print(f"Saved: debug\\v41_best_boxes.png and debug\\v41_best_mask.png")
print(f"Also saved top 5 candidates: v41_topXX_*")

cells_global = [(x0+x, y0+y, w, h) for x, y, w, h, _ in best["boxes"]]
with open("cells.txt", "w") as f:
    for x, y, w, h in sorted(cells_global, key=lambda t: (t[1], t[0])):
        f.write(f"{x},{y},{w},{h}\n")
print("Wrote cells.txt")
