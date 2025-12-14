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

S_MIN = 25
mask_color = ((S > S_MIN) * 255).astype(np.uint8)

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, k, iterations=2)
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN,  k, iterations=1)
save(OUTDIR / "v4_01_mask_color.png", mask_color)

cnts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise SystemExit("No ROI found. Lower S_MIN (e.g. 15).")

cnt = max(cnts, key=cv2.contourArea)
x, y, bw, bh = cv2.boundingRect(cnt)

pad = int(0.07 * max(bw, bh))
x0 = max(0, x - pad)
y0 = max(0, y - pad)
x1 = min(W, x + bw + pad)
y1 = min(H, y + bh + pad)

board_full = img[y0:y1, x0:x1].copy()
save(OUTDIR / "v4_02_board_roi.png", board_full)

# Aggressive tray crop: keep only the top part of ROI
CROP_BOTTOM = 0.80  # try 0.78–0.85
bh2, bw2 = board_full.shape[:2]
board = board_full[: int(bh2 * CROP_BOTTOM), :].copy()
save(OUTDIR / "v4_02b_board_roi_cropped.png", board)

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

        if w < 22 or h < 22:  # reject specks
            continue

        ar = w / float(h)
        if ar < 0.75 or ar > 1.33:
            continue

        # reject very thin junk
        if area < 250:
            continue

        boxes.append((x, y, w, h, area))
    return boxes

def score_boxes(boxes):
    # We want MANY similarly-sized squares
    n = len(boxes)
    if n < 8:
        return -1e9

    areas = np.array([b[4] for b in boxes], dtype=np.float32)
    sizes = np.array([(b[2] + b[3]) / 2.0 for b in boxes], dtype=np.float32)

    area_cv = float(areas.std() / (areas.mean() + 1e-6))
    size_cv = float(sizes.std() / (sizes.mean() + 1e-6))

    # Prefer 12–24 tiles typically visible
    target = 16.0
    count_penalty = abs(n - target) / target  # 0 at n=target

    # Higher is better:
    return (n / (1.0 + 3.0*area_cv + 2.0*size_cv)) * (1.0 / (1.0 + count_penalty))

# -----------------------------
# 2) Mask-based candidates (like v3, but tighter + more variety)
# -----------------------------
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
L = lab[:, :, 0]

mask_candidates = []

# Otsu + several blurs
for blur in [3, 5, 7]:
    Lb = cv2.GaussianBlur(L, (blur, blur), 0)
    _, m = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_candidates.append((f"otsu_bin_b{blur}", m))
    mask_candidates.append((f"otsu_inv_b{blur}", cv2.bitwise_not(m)))

# Adaptive + a few blocks
for block in [21, 31, 41]:
    for C in [-5, -9, -13]:
        Lb = cv2.GaussianBlur(L, (5, 5), 0)
        m = cv2.adaptiveThreshold(Lb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block, C)
        mask_candidates.append((f"adapt_bin_b{block}_C{C}", m))
        mask_candidates.append((f"adapt_inv_b{block}_C{C}", cv2.bitwise_not(m)))

morph_grid = [(0, 7), (0, 9), (5, 9), (7, 11), (9, 13)]

best = None

for name, base in mask_candidates:
    for ok, ck in morph_grid:
        m = cleanup(base, open_k=ok, close_k=ck)
        boxes = boxes_from_components(m)

        # Remove huge merged blobs by trimming top percentile
        if len(boxes) >= 8:
            areas = np.array([b[4] for b in boxes], dtype=np.float32)
            hi = np.percentile(areas, 92)
            boxes = [b for b in boxes if b[4] <= hi * 1.15]

        sc = score_boxes(boxes)
        if best is None or sc > best["score"]:
            best = {"mode": "mask", "name": name, "ok": ok, "ck": ck,
                    "mask": m, "boxes": boxes, "score": sc}

# -----------------------------
# 3) Edge+contour candidates (NEW)
# -----------------------------
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 9, 50, 50)

edges = cv2.Canny(gray, 40, 120)
edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
save(OUTDIR / "v4_03_edges.png", edges)

cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

edge_boxes = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 300:
        continue
    x, y, w, h = cv2.boundingRect(c)
    if w < 22 or h < 22:
        continue
    ar = w / float(h)
    if ar < 0.75 or ar > 1.33:
        continue

    # Fill ratio: rounded squares should have decent fill in bbox
    bbox_area = w * h
    fill = area / float(bbox_area + 1e-6)
    if fill < 0.35:
        continue

    edge_boxes.append((x, y, w, h, bbox_area))

# prune near-duplicates (NMS-ish)
edge_boxes = sorted(edge_boxes, key=lambda b: b[4], reverse=True)
kept = []
for b in edge_boxes:
    x, y, w, h, a = b
    bx1, by1 = x + w, y + h
    overlap = False
    for kx, ky, kw, kh, _ in kept:
        kx1, ky1 = kx + kw, ky + kh
        inter_w = max(0, min(bx1, kx1) - max(x, kx))
        inter_h = max(0, min(by1, ky1) - max(y, ky))
        inter = inter_w * inter_h
        union = (w*h) + (kw*kh) - inter + 1e-6
        if inter / union > 0.45:
            overlap = True
            break
    if not overlap:
        kept.append(b)

edge_boxes = [(x,y,w,h,a) for x,y,w,h,a in kept]
edge_sc = score_boxes(edge_boxes)

if best is None or edge_sc > best["score"]:
    best = {"mode": "edges", "name": "canny_contours",
            "mask": edges, "boxes": edge_boxes, "score": edge_sc}

# -----------------------------
# 4) Output best
# -----------------------------
dbg = board.copy()
for x, y, w, h, _ in best["boxes"]:
    cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)

save(OUTDIR / "v4_04_best_boxes.png", dbg)

print(f"ROI: x={x0}, y={y0}, w={x1-x0}, h={y1-y0} | crop_bottom={CROP_BOTTOM}")
print(f"BEST mode: {best['mode']} | {best.get('name')} | score={best['score']:.3f}")
print(f"Detected {len(best['boxes'])} boxes")
print(f"Saved: {OUTDIR/'v4_04_best_boxes.png'}")

# write global coords
cells_global = []
for x, y, w, h, _ in best["boxes"]:
    cells_global.append((x0 + x, y0 + y, w, h))

with open("cells.txt", "w") as f:
    for x, y, w, h in sorted(cells_global, key=lambda t: (t[1], t[0])):
        f.write(f"{x},{y},{w},{h}\n")

print("Wrote cells.txt (x,y,w,h in full screenshot coords)")
