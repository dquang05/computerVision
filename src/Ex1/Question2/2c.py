import numpy as np
import cv2

img = np.array([
    [30,  20,  30,  30],
    [30,  30, 100, 110],
    [30,  30,  30, 150],
    [30, 230, 240, 250]
], dtype=np.uint8)

T_otsu, bin_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Image matrix =")
print(img)
print("Otsu threshold T =", T_otsu)
print("Otsu threshold result =")
print(bin_otsu)