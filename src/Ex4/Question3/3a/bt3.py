import cv2
import numpy as np

# Tọa độ 4 điểm ở Frame 1 (x, y) - Điểm 1, 2, 6, 10
pts_frame1 = np.float32([
    [5, 5], 
    [6, 5], 
    [8, 7], 
    [5, 6]
])

# Tọa độ 4 điểm tương ứng ở Frame 2 (x, y)
pts_frame2 = np.float32([
    [3, 7], 
    [3, 6], 
    [5, 4], 
    [4, 7]
])

print("--- 1. Kiểm tra Ma trận 2D (Affine) ---")
# Cần 3 điểm cho Affine Transform
matrix_2d = cv2.getAffineTransform(pts_frame1[:3], pts_frame2[:3])
print(np.round(matrix_2d, 2))

print("\n--- 2. Kiểm tra Ma trận Homography ---")
# Cần ít nhất 4 điểm cho Homography
matrix_homography, status = cv2.findHomography(pts_frame1, pts_frame2)
print(np.round(matrix_homography, 2))