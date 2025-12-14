import cv2
import numpy as np
from pathlib import Path

# ---------- helpers ----------

def largest_cc_bbox(mask: np.ndarray):
    """Return bbox (x,y,w,h) of largest connected component in a binary mask."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return None
    # skip background idx 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    x, y, w, h, _ = stats[i]
    return (x, y, w, h)

def find_peaks_1d(v, min_dist=30, rel_thresh=0.35):
    """
    Simple peak finder:
      - smooth
      - take local maxima above rel_thresh * max
      - enforce min_dist by greedy selection
    Returns sorted peak indices.
    """
    v = v.astype(np.float32)
    if v.size < 5:
        return np.array([], dtype=int)

    # smooth with 1D gaussian (implemented as 2D blur)
    v2 = cv2.GaussianBlur(v.reshape(1, -1), (1, 0), sigmaX=0, sigmaY=0)  # no-op safe
    v2 = cv2.GaussianBlur(v.reshape(1, -1), (1, 21), sigmaX=0, sigmaY=0).ravel()

    mx = float(v2.max())
    if mx <= 0:
        return np.array([], dtype=int)
    thr = mx * rel_thresh

    # local maxima
    cand = np.where((v2 > thr) &
                    (v2 >= np.roll(v2, 1)) &
                    (v2 >= np.roll(v2, -1)))[0]
    if cand.size == 0:
        return np.array([], dtype=int)

    # greedy enforce min_dist (keep strongest first)
    strengths = v2[cand]
    order = cand[np.argsort(-strengths)]
    kept = []
    for idx in order:
        if all(abs(idx - k) >= min_dist for k in kept):
            kept.append(int(idx))
    return np.array(sorted(kept), dtype=int)

def dedupe_nearby(peaks, eps=8):
    """Merge peaks that are very close by snapping to their mean."""
    if len(peaks) == 0:
        return peaks
    peaks = np.array(sorted(peaks))
    groups = [[int(peaks[0])]]
    for p in peaks[1:]:
        if abs(int(p) - groups[-1][-1]) <= eps:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    merged = [int(round(np.mean(g))) for g in groups]
    return np.array(sorted(merged), dtype=int)

# ---------- main pipeline ----------

def extract_cells_from_screenshot(
    img_path: str,
    out_dir: str = "debug_gridlines",
    force_roi=None,          # (x,y,w,h) in full image coords if you want to override autodetect
    lower_half_only=True,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read: {img_path}")

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # ---- 1) find board ROI ----
    if force_roi is not None:
        x, y, w, h = force_roi
    else:
        y_start = H // 2 if lower_half_only else 0
        # saturation mask: board colors pop vs dark bg
        sat_mask = (sat > 35) & (val > 35)
        sat_mask[:y_start, :] = False

        # clean up a bit
        m = sat_mask.astype(np.uint8) * 255
        m = cv2.medianBlur(m, 5)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        bbox = largest_cc_bbox(m > 0)
        if bbox is None:
            raise RuntimeError("ROI autodetect failed (no connected component). Try force_roi.")
        x, y, w, h = bbox

        # pad ROI
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(W - x, w + 2 * pad)
        h = min(H - y, h + 2 * pad)

    roi = img_bgr[y:y+h, x:x+w].copy()
    cv2.imwrite(str(out / "01_roi.png"), roi)

    # ---- 2) edges + projections to get grid lines ----
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # a little blur helps dashed borders turn into “line energy”
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    edges = cv2.Canny(roi_blur, 40, 140)
    cv2.imwrite(str(out / "02_edges.png"), edges)

    # projections (sum of edge pixels per column/row)
    proj_x = edges.sum(axis=0).astype(np.float32)
    proj_y = edges.sum(axis=1).astype(np.float32)

    # peaks = likely grid boundaries
    px = find_peaks_1d(proj_x, min_dist=40, rel_thresh=0.30)
    py = find_peaks_1d(proj_y, min_dist=40, rel_thresh=0.30)

    px = dedupe_nearby(px, eps=10)
    py = dedupe_nearby(py, eps=10)

    # If you’re seeing too few lines, loosen thresholds:
    if len(px) < 5 or len(py) < 4:
        px = dedupe_nearby(find_peaks_1d(proj_x, min_dist=30, rel_thresh=0.22), eps=10)
        py = dedupe_nearby(find_peaks_1d(proj_y, min_dist=30, rel_thresh=0.22), eps=10)

    # draw lines for debug
    dbg = roi.copy()
    for p in px:
        cv2.line(dbg, (int(p), 0), (int(p), h-1), (0, 255, 0), 2)
    for p in py:
        cv2.line(dbg, (0, int(p)), (w-1, int(p)), (0, 255, 0), 2)
    cv2.imwrite(str(out / "03_grid_lines.png"), dbg)

    if len(px) < 2 or len(py) < 2:
        raise RuntimeError(f"Not enough grid lines found. px={len(px)}, py={len(py)}. Try force_roi or tweak thresholds.")

    # ---- 3) build candidate rectangles between adjacent lines ----
    # Ensure we include ROI borders as “lines” if needed
    px2 = np.unique(np.clip(np.concatenate(([0], px, [w-1])), 0, w-1))
    py2 = np.unique(np.clip(np.concatenate(([0], py, [h-1])), 0, h-1))

    px2.sort()
    py2.sort()

    # ---- 4) filter candidates by interior brightness (real cells are lighter than background hole) ----
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_val = roi_hsv[:, :, 2]

    cells = []
    vis = roi.copy()

    for yi in range(len(py2)-1):
        for xi in range(len(px2)-1):
            x0, x1 = int(px2[xi]), int(px2[xi+1])
            y0, y1 = int(py2[yi]), int(py2[yi+1])

            bw = x1 - x0
            bh = y1 - y0
            if bw < 40 or bh < 40:
                continue

            # sample central area (avoid dashed borders)
            pad = int(min(bw, bh) * 0.18)
            cx0, cx1 = x0 + pad, x1 - pad
            cy0, cy1 = y0 + pad, y1 - pad
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            patch_v = roi_val[cy0:cy1, cx0:cx1]
            mean_v = float(patch_v.mean())
            std_v = float(patch_v.std())

            # heuristic: real “cell squares” are brighter AND have some texture/variance
            if mean_v > 55 and std_v > 6:
                # keep in full screenshot coords
                full = (x + x0, y + y0, bw, bh)
                cells.append(full)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.imwrite(str(out / "04_cells.png"), vis)

    # write cells
    with open("cells.txt", "w", encoding="utf-8") as f:
        for (fx, fy, fw, fh) in cells:
            f.write(f"{fx},{fy},{fw},{fh}\n")

    print(f"ROI: x={x}, y={y}, w={w}, h={h}")
    print(f"Grid lines: px={len(px)} py={len(py)} | candidates kept: {len(cells)}")
    print(f"Saved debug to: {out.resolve()}")
    print("Wrote cells.txt (x,y,w,h in full screenshot coords)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Full screenshot image path")
    ap.add_argument("--out", default="debug_gridlines")
    ap.add_argument("--roi", default=None, help="Force ROI as x,y,w,h (full image coords)")
    args = ap.parse_args()

    roi = None
    if args.roi:
        parts = [int(p.strip()) for p in args.roi.split(",")]
        if len(parts) != 4:
            raise ValueError("--roi must be x,y,w,h")
        roi = tuple(parts)

    extract_cells_from_screenshot(args.image, out_dir=args.out, force_roi=roi)
