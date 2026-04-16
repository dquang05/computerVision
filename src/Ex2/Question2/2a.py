import cv2
import numpy as np

img = np.array([
    [12,  2, 18,  4],
    [ 8,  6, 10, 11],
    [ 9, 16, 15, 18],
    [ 1,  6,  3, 30]
], dtype=np.float32)

templ = np.array([
    [ 6, 10, 11],
    [16, 15, 18],
    [ 2,  3, 30]
], dtype=np.float32)

result = cv2.matchTemplate(img, templ, cv2.TM_CCORR_NORMED)

print("NCC result matrix:")
print(result)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print("\nLargest NCC value:", max_val)

# Convert to G(row, col) coordinates with row and col starting from 1
row = max_loc[1] + 1
col = max_loc[0] + 1

print(f"Best matching position is: G({row},{col})")