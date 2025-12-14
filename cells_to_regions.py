import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import json

IMG = "IMG_2050.png"

img = cv2.imread(IMG)
cells = []

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if 40 < w < 120 and abs(w-h) < 15:
        cx, cy = x+w//2, y+h//2
        cells.append((x,y,w,h,cx,cy))

# Sort into rows then cols
cells.sort(key=lambda c: (c[5], c[4]))

# Sample colors
colors = []
coords = []
for x,y,w,h,cx,cy in cells:
    patch = img[y+10:y+h-10, x+10:x+w-10]
    avg = patch.mean(axis=(0,1))
    colors.append(avg)
    coords.append((cx,cy))

colors = np.array(colors)

# Guess number of regions (works well here)
k = 6
kmeans = KMeans(n_clusters=k, random_state=0).fit(colors)
labels = kmeans.labels_

regions = defaultdict(list)
for (cx,cy), lab in zip(coords, labels):
    regions[int(lab)].append((cx,cy))

out = {
    "regions": [
        {"id": i, "cells": v, "color": kmeans.cluster_centers_[i].tolist()}
        for i,v in regions.items()
    ]
}

with open("regions_raw.json","w") as f:
    json.dump(out,f,indent=2)

print("Wrote regions_raw.json")
