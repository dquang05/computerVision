import cv2
import numpy as np
import math
from pathlib import Path

# 1. Cấu hình thông số hiệu chuẩn (K, dist từ kết quả calib)
K = np.array([
    [3.84348896e+03, 0.00000000e+00, 1.00263233e+03],
    [0.00000000e+00, 3.84560121e+03, 1.19370403e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# Đã sửa lại cú pháp khai báo dist_coeffs
dist_coeffs = np.array([
    [1.73646595e-01, -8.46663554e-01, -4.82888456e-03, 3.47256356e-03, 3.74947528e+00]
])

B = 68  # Baseline (mm) theo yêu cầu

# Resolve image paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
LEFT_IMAGE_PATH = BASE_DIR / "a.jpg"
RIGHT_IMAGE_PATH = BASE_DIR / "b.jpg"

# Lưu trữ điểm click
pts_left = []
pts_right = []

def mouse_callback_left(event, x, y, flags, param):
    # Nâng giới hạn click lên 3 điểm
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_left) < 3:
        pts_left.append((x, y))
        cv2.circle(imgL_display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(imgL_display, f"P{len(pts_left)}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
        cv2.imshow("Left Image", imgL_display)

def mouse_callback_right(event, x, y, flags, param):
    # Nâng giới hạn click lên 3 điểm
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_right) < 3:
        pts_right.append((x, y))
        cv2.circle(imgR_display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(imgR_display, f"P{len(pts_right)}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
        cv2.imshow("Right Image", imgR_display)

def get_3d_point(uL, vL, uR, vR, K, B):
    """Tính tọa độ 3D theo công thức Stereo"""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    disparity = uL - uR
    if abs(disparity) < 0.1:
        return None, disparity
    
    Z = (fx * B) / disparity
    X = (uL - cx) * Z / fx
    Y = (vL - cy) * Z / fy
    return np.array([X, Y, Z]), disparity

def compute_orientation(P1, P2, P3):
    """Tính ma trận xoay R và các góc Euler (Roll, Pitch, Yaw) từ 3 điểm 3D"""
    # Vector từ P1 đến P2 (Trục X cục bộ)
    v1 = P2 - P1
    v1 = v1 / np.linalg.norm(v1)

    # Vector từ P1 đến P3
    v2_temp = P3 - P1

    # Trục Z cục bộ (Vuông góc với mặt phẳng P1-P2-P3)
    v3 = np.cross(v1, v2_temp)
    if np.linalg.norm(v3) < 1e-6:
        raise ValueError("3 điểm bị thẳng hàng, không thể tạo thành mặt phẳng!")
    v3 = v3 / np.linalg.norm(v3)

    # Trục Y cục bộ (Vuông góc với X và Z)
    v2 = np.cross(v3, v1)
    v2 = v2 / np.linalg.norm(v2)

    # Ghép 3 vector thành ma trận R
    R_local = np.column_stack([v1, v2, v3])

    # Chuẩn hóa SVD (Singular Value Decomposition) để khử nhiễu
    U, S, Vt = np.linalg.svd(R_local)
    R = U @ Vt

    # Tính các góc Euler theo quy ước ZYX
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Xử lý trường hợp Gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    # Đổi sang độ (degrees)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw), R

# 2. Đọc và tiền xử lý ảnh
imgL_raw = cv2.imread(str(LEFT_IMAGE_PATH))
imgR_raw = cv2.imread(str(RIGHT_IMAGE_PATH))

if imgL_raw is None or imgR_raw is None:
    print("Lỗi đọc file ảnh! Vui lòng kiểm tra lại đường dẫn.")
else:
    # Khử méo ảnh (Undistort) trước khi chọn điểm
    imgL = cv2.undistort(imgL_raw, K, dist_coeffs)
    imgR = cv2.undistort(imgR_raw, K, dist_coeffs)
    
    imgL_display = imgL.copy()
    imgR_display = imgR.copy()

        # Khai báo cửa sổ với thuộc tính WINDOW_NORMAL cho phép resize
    cv2.namedWindow("Left Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Image", cv2.WINDOW_NORMAL)
        
    # Đặt kích thước cửa sổ mặc định (Chiều rộng, Chiều cao)
    # Có thể thay đổi số 800, 600 này cho vừa với màn hình
    cv2.resizeWindow("Left Image", 900, 700)
    cv2.resizeWindow("Right Image", 900, 700)
    # Gắn hàm callback xử lý click chuột
    cv2.setMouseCallback("Left Image", mouse_callback_left)
    cv2.setMouseCallback("Right Image", mouse_callback_right)

    print("Bước 1: Click 3 điểm (P1, P2, P3) trên ảnh Trái, sau đó click 3 điểm tương ứng trên ảnh Phải.")
    print("Lưu ý: P1->P2 sẽ định hình trục X, P1->P2->P3 định hình mặt phẳng.")
    print("Nhấn 'q' sau khi đã chọn đủ 6 điểm (3 cặp).")

    while True:
            cv2.imshow("Left Image", imgL_display)
            cv2.imshow("Right Image", imgR_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    # 3. Tính toán kết quả
    if len(pts_left) == 3 and len(pts_right) == 3:
        # Tính tọa độ 3D và lấy cả giá trị disparity cho 3 điểm
        P1_result = get_3d_point(pts_left[0][0], pts_left[0][1], pts_right[0][0], pts_right[0][1], K, B)
        P2_result = get_3d_point(pts_left[1][0], pts_left[1][1], pts_right[1][0], pts_right[1][1], K, B)
        P3_result = get_3d_point(pts_left[2][0], pts_left[2][1], pts_right[2][0], pts_right[2][1], K, B)

        if P1_result[0] is not None and P2_result[0] is not None and P3_result[0] is not None:
            P1_3d, d1 = P1_result
            P2_3d, d2 = P2_result
            P3_3d, d3 = P3_result

            # 3.1 Tính khoảng cách P1-P2 và P1-P3
            length_L = np.linalg.norm(P2_3d - P1_3d)
            length_P13 = np.linalg.norm(P3_3d - P1_3d)
            
            # --- IN KẾT QUẢ ĐO 3D (GIỐNG HÌNH MẪU) ---
            print("\n" + "="*60)
            print("KẾT QUẢ ĐO 3D")
            print("="*60)
            print(f"P1 (3D): X={P1_3d[0]:.4f} mm, Y={P1_3d[1]:.4f} mm, Z={P1_3d[2]:.4f} mm")
            print(f"P2 (3D): X={P2_3d[0]:.4f} mm, Y={P2_3d[1]:.4f} mm, Z={P2_3d[2]:.4f} mm")
            print(f"P3 (3D): X={P3_3d[0]:.4f} mm, Y={P3_3d[1]:.4f} mm, Z={P3_3d[2]:.4f} mm")
            print()
            print(f"Disparity P1: {d1:.2f} pixels")
            print(f"Disparity P2: {d2:.2f} pixels")
            print(f"Disparity P3: {d3:.2f} pixels")
            print()
            print(f"Chiều dài đoạn nối P1-P2: L = {length_L:.4f} mm")
            print(f"Chiều dài đoạn nối P1-P3: = {length_P13:.4f} mm")
            print("="*60)
            
            # 3.2 Tính và in R-P-Y (Nâng cao)
            try:
                roll, pitch, yaw, R_matrix = compute_orientation(P1_3d, P2_3d, P3_3d)
                
                print("\n" + "="*60)
                print("GÓC EULER (ZYX convention):")
                print("="*60)
                print(f"  Roll  (quay quanh X):   {roll:+.2f}°")
                print(f"  Pitch (quay quanh Y):   {pitch:+.2f}°")
                print(f"  Yaw   (quay quanh Z):   {yaw:+.2f}°")
                print("="*60)
                
            except Exception as e:
                print(f"\nLỗi khi tính toán góc: {e}")
        else:
            print("\nLỗi: Disparity không hợp lệ (có điểm bị chia cho 0).")
    else:
        print(f"\nChưa chọn đủ số điểm! Trái: {len(pts_left)}/3, Phải: {len(pts_right)}/3.")