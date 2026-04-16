import cv2
import numpy as np
import math

matrix_c = np.array([
    [0,   0,   0,   255, 0,   0,   0],
    [0,   0,   255, 255, 255, 0,   0],
    [0,   255, 255, 255, 255, 255, 0],
    [0,   0,   255, 255, 255, 0,   0],
    [0,   0,   0,   255, 255, 0,   0],
    [0,   0,   0,   255, 255, 0,   0],
    [0,   0,   0,   255, 0,   0,   0]
], dtype=np.uint8)

matrix_T = matrix_c.T
M = cv2.moments(matrix_T)

# TỌA ĐỘ TRỌNG TÂM
cX = M['m10'] / M['m00'] + 1
cY = M['m01'] / M['m00'] + 1
print(f"2. Tọa độ tâm (xc, yc): ({cX:.2f}, {cY:.2f})")
# DIỆN TÍCH
# Diện tích M0,0 theo OpenCV (tổng giá trị pixel)
m00 = M['m00']
# Số lượng pixel
area_pixels = m00 / 255
print(f"1. Diện tích (M00): {m00} -> {area_pixels} pixels")
# CHU VI
contours, _ = cv2.findContours(matrix_T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
perimeter = cv2.arcLength(contours[0], True)
print(f"4. Chu vi (Contours Perimeter): {perimeter:.2f}")
# C. GÓC XOAY
mu20 = M['mu20']
mu02 = M['mu02']
mu11 = M['mu11']
angle_rad = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
angle_deg = math.degrees(angle_rad)
print(f"3. Góc xoay (Angle): {angle_deg:.2f} độ")