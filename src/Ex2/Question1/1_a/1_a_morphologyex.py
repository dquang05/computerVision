import cv2
import numpy as np

A = np.array([
    [0,   0,   0,   0,   0,   0,   0],
    [0,   0, 255, 255, 255,   0,   0],
    [0, 255, 255, 255, 255, 255,   0],
    [0,   0, 255,   0, 255,   0,   0],
    [0,   0, 255,   0, 255,   0,   0],
    [0,   0, 255, 255, 255,   0,   0],
    [0,   0,   0,   0,   0,   0,   0]
], dtype=np.uint8)

kernel = np.array([
    [ 1,  0,  1],
    [ 0,  0,  0],
    [ 0, -1,  0]
], dtype=np.int8)

H = cv2.morphologyEx(
    src=A,
    op=cv2.MORPH_HITMISS,
    kernel=kernel,
    dst=None,
    anchor=(-1, -1),
    iterations=1,
    borderType=cv2.BORDER_REPLICATE
)

print("Result")
print(H)
