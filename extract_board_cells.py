import cv2
import numpy as np
from pathlib import Path

IMG = "IMG_2050.png"
OUT = "debug_cells.png"

img = cv2.imread(IMG)
if img is None:
    raise SystemExit("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Edge detect
edges = cv2.Canny(blur, 50, 150)

# Find contours
cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cells = []
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if 40 < w < 120 and abs(w-h) < 15:   # tuned for this UI
        cells.append((x,y,w,h))

# Draw detected cells
dbg = img.copy()
for x,y,w,h in cells:
    cv2.rectangle(dbg,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imwrite(OUT, dbg)
print(f"Detected {len(cells)} cells → {OUT}")
