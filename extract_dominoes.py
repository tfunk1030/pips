import cv2
import numpy as np

IMG = "IMG_2050.png"

img = cv2.imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dominoes = []
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if 120 < w < 240 and 50 < h < 100:
        tile = img[y:y+h, x:x+w]
        left = tile[:, :w//2]
        right = tile[:, w//2:]

        def count_pips(half):
            g = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
            _,t = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY_INV)
            cnts,_ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return sum(1 for c in cnts if 10 < cv2.contourArea(c) < 200)

        a = count_pips(left)
        b = count_pips(right)
        dominoes.append((a,b))

print("Dominoes:", dominoes)
