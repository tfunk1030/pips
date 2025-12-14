import cv2
import numpy as np
from pathlib import Path

IMG = "IMG_2050.png"   # set this
OUTDIR = Path("debug")
OUTDIR.mkdir(exist_ok=True)

img = cv2.imread(IMG)
if img is None:
    raise SystemExit(f"Could not read image: {IMG}")

H, W = img.shape[:2]

def save(path, im):
    cv2.imwrite(str(path), im)

# -----------------------------
# 1) Board ROI via saturation
# -----------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
_, S, _ = cv2.split(hsv)

S_MIN = 25  # a bit lower than before; board pastels can be low sat
mask_color = (S > S_MIN).astype(np.uint8) * 255

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, k, iterations=2)
mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN,  k, iterations=1)

save(OUTDIR / "v3_01_mask_color.png", mask_color)

cnts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise SystemExit("No board-ish region found. Try lowering S_MIN to ~15.")

cnt = max(cnts, key=cv2.contourArea)
x, y, bw, bh = cv2.boundingRect(cnt)

pad = int(0.07 * max(bw, bh))
x0 = max(0, x - pad)
y0 = max(0, y - pad)
x1 = min(W, x + bw + pad)
y1 = min(H, y + bh + pad)

board = img[y0:y1, x0:x1].copy()
save(OUTDIR / "v3_02_board_roi.png", board)

# Optional: crop out the domino tray area if ROI accidentally includes it.
# The board tiles are above the tray; cropping bottom 15% is usually safe.
bh2, bw2 = board.shape[:2]
board_crop = board[: int(bh2 * 0.88), :]  # keep top 88%
save(OUTDIR / "v3_02b_board_roi_cropped.png", board_crop)

# We'll run detection on board_crop, but write global coords correctly.
board = board_crop
bh2, bw2 = board.shape[:2]

# -----------------------------
# 2) Candidate mask generators
# -----------------------------
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

def gen_masks():
    masks = []

    # (A) Otsu on L (bright tiles vs dark interior/edges)
    for blur in [3, 5, 7]:
        Lb = cv2.GaussianBlur(L, (blur, blur), 0)
        _, m = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(("otsu_bin_blur"+str(blur), m))
        masks.append(("otsu_inv_blur"+str(blur), cv2.bitwise_not(m)))

    # (B) Adaptive thresholds
    for block in [21, 31, 41]:
        for C in [-3, -7, -11]:
            Lb = cv2.GaussianBlur(L, (5, 5), 0)
            m = cv2.adaptiveThreshold(Lb, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      block, C)
            masks.append((f"adapt_bin_b{block}_C{C}", m))
            masks.append((f"adapt_inv_b{block}_C{C}", cv2.bitwise_not(m)))

    # (C) “Chroma” (distance from neutral in AB) to emphasize colored regions/borders
    # Pastel tiles still have some chroma; the hole/background is more neutral.
    ab = np.sqrt((A.astype(np.float32)-128.0)**2 + (B.astype(np.float32)-128.0)**2)
    ab = cv2.normalize(ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ab_blur = cv2.GaussianBlur(ab, (5, 5), 0)
    _, m = cv2.threshold(ab_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(("chroma_otsu", m))
    masks.append(("chroma_otsu_inv", cv2.bitwise_not(m)))

    return masks

def cleanup(mask, open_k, close_k, open_it=1, close_it=2):
    m = mask.copy()
    if open_k > 0:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1, iterations=open_it)
    if close_k > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=close_it)
    return m

def score_components(mask):
    # connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]

    # Remove tiny junk by percentile cut
    lo = np.percentile(areas, 20)
    hi = np.percentile(areas, 95)

    cand = []
    for i in range(len(areas)):
        area = areas[i]
        if area < max(60, lo):  # avoid specks
            continue
        if area > hi * 1.2:     # avoid huge merged blobs
            continue

        w = ws[i]; h = hs[i]
        if w < 18 or h < 18:
            continue

        ar = w / float(h)
        if ar < 0.72 or ar > 1.38:
            continue

        cand.append((xs[i], ys[i], w, h, area))

    if len(cand) < 6:
        return None

    # cells should have similar area; use coefficient of variation (lower is better)
    a = np.array([c[4] for c in cand], dtype=np.float32)
    cv = float(a.std() / (a.mean() + 1e-6))

    # prefer a reasonable count (often ~14-ish cells visible, but allow range)
    n = len(cand)
    count_score = 0.0
    if 10 <= n <= 30:
        count_score = 1.0
    elif 6 <= n <= 40:
        count_score = 0.6
    else:
        count_score = 0.2

    # final score: more candidates + uniformity
    score = (count_score * n) * (1.0 / (1.0 + 3.0 * cv))
    return score, cand, cv, n

# -----------------------------
# 3) Autotune + pick best
# -----------------------------
best = None
best_name = None
best_mask = None

masks = gen_masks()

# try a small grid of morph cleanup
morph_params = [
    (0, 9),   # only close
    (5, 9),   # open then close
    (7, 11),
    (9, 13),
]

for name, base in masks:
    for open_k, close_k in morph_params:
        m = cleanup(base, open_k=open_k, close_k=close_k)
        res = score_components(m)
        if res is None:
            continue
        score, cand, cv, n = res
        if (best is None) or (score > best[0]):
            best = (score, cand, cv, n, open_k, close_k)
            best_name = name
            best_mask = m

if best is None:
    save(OUTDIR / "v3_FAIL_last_mask.png", masks[0][1])
    raise SystemExit("Autotune failed to find a usable tile mask. Check v3_02_board_roi_cropped.png and we’ll adjust ROI/mask logic.")

score, cand, cv, n, open_k, close_k = best
save(OUTDIR / "v3_03_best_mask.png", best_mask)

dbg = board.copy()
for x, y, ww, hh, area in cand:
    cv2.rectangle(dbg, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
save(OUTDIR / "v3_04_best_cells.png", dbg)

print(f"Board ROI: x={x0}, y={y0}, w={x1-x0}, h={y1-y0} (cropped top 88%)")
print(f"BEST mask: {best_name} | morph open={open_k} close={close_k}")
print(f"Detected {n} candidates | area CV={cv:.3f} | score={score:.2f}")
print(f"Saved: {OUTDIR/'v3_03_best_mask.png'} and {OUTDIR/'v3_04_best_cells.png'}")

# Write global coords (remember we cropped board height)
cells_global = []
for x, y, ww, hh, area in cand:
    cells_global.append((x0 + x, y0 + y, ww, hh))

with open("cells.txt", "w") as f:
    for x, y, ww, hh in sorted(cells_global, key=lambda t: (t[1], t[0])):
        f.write(f"{x},{y},{ww},{hh}\n")

print("Wrote cells.txt (x,y,w,h in full screenshot coords)")
