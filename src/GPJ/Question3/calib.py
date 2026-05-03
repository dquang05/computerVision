import numpy as np
import cv2
from pathlib import Path

# 1. CẤU HÌNH THÔNG SỐ (CẦN KIỂM TRA KỸ)
# Số GÓC GIAO NHAU bên trong bàn cờ (Ví dụ bàn cờ 10x7 ô thì điền 9, 6)
CHECKERBOARD = (7, 10) 

# Kích thước thực tế của 1 cạnh ô vuông (Đơn vị: mm)
SQUARE_SIZE = 25.0 

# Đường dẫn đến thư mục chứa 25 ảnh (tương đối theo file này)
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "25image"

# 2. CHUẨN BỊ MẢNG LƯU TRỮ
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = [] # Điểm 3D thực tế
imgpoints = [] # Điểm 2D trên ảnh

# 3. TRÍCH XUẤT GÓC TỪ 25 ẢNH
images = sorted(IMAGE_FOLDER.glob("*.jpg"))
if len(images) == 0:
    print(f"Lỗi: Không tìm thấy ảnh nào! Hãy kiểm tra lại xem ảnh là đuôi .jpg hay .png nhé.")
    exit()

print(f"Đã tìm thấy {len(images)} ảnh. Đang tiến hành quét góc...")
valid_images = 0

for fname in images:
    img = cv2.imread(str(fname))
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tìm góc bàn cờ (Tích hợp các cờ lọc nhiễu tự động)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_FAST_CHECK + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE) # type: ignore
    
    if ret is True:
        objpoints.append(objp)
        valid_images += 1
        
        # Tinh chỉnh tọa độ góc xuống điểm ảnh phụ
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Hiển thị kết quả bắt góc (bấm phím bất kỳ để xem ảnh tiếp theo, ESC để bỏ qua xem)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Kiem tra bat goc', cv2.resize(img, (800, 600)))
        key = cv2.waitKey(100)
        if key == 27: 
            cv2.destroyAllWindows()

cv2.destroyAllWindows()
print(f"Bắt góc thành công trên {valid_images}/{len(images)} ảnh.")

# 4. TÍNH TOÁN CALIBRATION
if valid_images < 10:
    print("Cảnh báo: Số lượng ảnh hợp lệ quá ít (dưới 10 ảnh), ma trận có thể không chính xác!")

print("\nĐang tính toán ma trận K, R, t...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # type: ignore

# 5. ĐÁNH GIÁ SAI SỐ (QUAN TRỌNG NHẤT)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

total_error = mean_error / len(objpoints)

print("\n" + "="*50)
print("KẾT QUẢ CALIBRATION")
print("="*50)
print(f"Sai số tái chiếu : {total_error:.4f} pixels")
if total_error > 1.0:
    print(" -> [KHÔNG ỔN]: Sai số đang > 1.0. Nên chụp lại bộ 25 ảnh khác rõ nét và đa dạng góc hơn.")
else:
    print(" -> [RẤT TỐT]: Sai số < 1.0. Bộ thông số này rất chuẩn xác.")

print("\n1. Ma trận nội tại K (Copy ma trận này để dùng):")
print(mtx)

print("\n2. Hệ số biến dạng dist (k1, k2, p1, p2, k3):")
print(dist)

# Trích xuất R và t cho bức ảnh ĐẦU TIÊN (Ảnh index 0)
R_first, _ = cv2.Rodrigues(rvecs[0])
print("\n3. Ma trận xoay R (của ảnh đầu tiên):")
print(R_first)

print("\n4. Vector tịnh tiến t [mm] (của ảnh đầu tiên):")
print(tvecs[0])