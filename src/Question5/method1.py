import cv2 as cv
import numpy as np
import random

def phan_loai_xoai(h_value):

    if h_value < 52:
        return "CHIN (Ripe)"
    elif 52 <= h_value < 70:
        return "VUA (Medium)"
    else:
        return "SONG (Unripe)"

def lay_mau_va_tinh_hue(img, mask, num_points=3):
    # Tìm tất cả điểm thuộc vùng xoài
    points_mango = cv.findNonZero(mask)

    # Kiểm tra xem có đủ điểm để random không
    if points_mango is None or len(points_mango) < num_points:
        print("Lỗi: Không tìm thấy vùng xoài hoặc vùng quá nhỏ!")
        return None, None

    # Chọn ngẫu nhiên các điểm
    random_indices = random.sample(range(len(points_mango)), num_points)
    
    display_img = img.copy()
    total_hue = 0 
    
    print(f"{'Điểm':<5} | {'R,G,B':<15} | {'Hue (Standard)':<15}")
    print("-" * 40)

    for i, idx in enumerate(random_indices):
        x, y = points_mango[idx][0]
        
        # 1. Lấy màu BGR từ ảnh gốc
        b, g, r = img[y, x]
        
        # 2. Tính Hue từ RGB
        pixel_bgr = np.uint8([[[b, g, r]]]) 
        pixel_hsv = cv.cvtColor(pixel_bgr, cv.COLOR_BGR2HSV)
        
        # --- SỬA Ở ĐÂY: Nhân 2 để ra hệ màu 360 độ ---
        h_val = pixel_hsv[0][0][0] * 2  
        
        total_hue += h_val 
        
        # In thông tin
        print(f"#{i+1:<4} | {r},{g},{b:<11} | {h_val}")

        # Vẽ lên ảnh hiển thị
        cv.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv.putText(display_img, f"{r},{g},{b}", (x+10, y), 
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Tính trung bình
    avg_hue = total_hue / num_points
    return avg_hue, display_img
    # 1. Đọc ảnh
img_original1 = cv.imread('Chin.jpg', cv.IMREAD_COLOR) 
img_original2 = cv.imread('Medium.jpg', cv.IMREAD_COLOR) 
img_original3 = cv.imread('Xanh.jpg', cv.IMREAD_COLOR) 
img1 = cv.resize(img_original1, (500, 500))
img2 = cv.resize(img_original2, (500, 500))
img3 = cv.resize(img_original3, (500, 500))


    # --- BƯỚC 1: TÁCH NỀN ---
hsv_img1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
hsv_img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
hsv_img3 = cv.cvtColor(img3, cv.COLOR_BGR2HSV)

    # Mask nền xanh dương
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([140, 255, 255])
mask_background1 = cv.inRange(hsv_img1, lower_blue, upper_blue)
mask_background2 = cv.inRange(hsv_img2, lower_blue, upper_blue)
mask_background3 = cv.inRange(hsv_img3, lower_blue, upper_blue)

    # Đảo ngược mask lấy quả xoài
mask_mango1 = cv.bitwise_not(mask_background1)
mask_mango2= cv.bitwise_not(mask_background2)
mask_mango3= cv.bitwise_not(mask_background3)
kernel = np.ones((5, 5), np.uint8)
mask_mango1 = cv.morphologyEx(mask_mango1, cv.MORPH_OPEN, kernel)
mask_mango1 = cv.morphologyEx(mask_mango1, cv.MORPH_CLOSE, kernel)
mask_mango2 = cv.morphologyEx(mask_mango2, cv.MORPH_OPEN, kernel)
mask_mango2 = cv.morphologyEx(mask_mango2, cv.MORPH_CLOSE, kernel)
mask_mango3 = cv.morphologyEx(mask_mango3, cv.MORPH_OPEN, kernel)
mask_mango3 = cv.morphologyEx(mask_mango3, cv.MORPH_CLOSE, kernel)

    # --- BƯỚC 2: XỬ LÝ 3 ĐIỂM ---
points_mango1 = cv.findNonZero(mask_mango1)
points_mango2= cv.findNonZero(mask_mango2)
points_mango3 = cv.findNonZero(mask_mango3)


    # --- BƯỚC 3: TÍNH TOÁN (Gọi hàm cho từng ảnh) ---
hue_trung_binh1, anh_ket_qua1 = lay_mau_va_tinh_hue(img1, mask_mango1, num_points=3)
hue_trung_binh2, anh_ket_qua2 = lay_mau_va_tinh_hue(img2, mask_mango2, num_points=3)
hue_trung_binh3, anh_ket_qua3 = lay_mau_va_tinh_hue(img3, mask_mango3, num_points=3)

    # --- BƯỚC 4: HIỂN THỊ KẾT QUẢ ---
font = cv.FONT_HERSHEY_SIMPLEX

    # --- Xử lý hiển thị Ảnh 1 ---
if anh_ket_qua1 is not None:
        # 1. Phân loại
    ket_luan1 = phan_loai_xoai(hue_trung_binh1)
        
        # 2. Vẽ lên ảnh kết quả 1 (anh_ket_qua1)
    cv.rectangle(anh_ket_qua1, (0, 0), (300, 70), (0, 0, 0), -1) 
    cv.putText(anh_ket_qua1, f"H_avg: {hue_trung_binh1:.1f}", (10, 25), font, 0.7, (0, 255, 255), 2)
    cv.putText(anh_ket_qua1, ket_luan1, (10, 55), font, 0.7, (0, 255, 0), 2)
        
        # 3. Show cửa sổ
    cv.imshow("Ket qua 1 (Chin)", anh_ket_qua1)

    # --- Xử lý hiển thị Ảnh 2 ---
    if anh_ket_qua2 is not None:
        ket_luan2 = phan_loai_xoai(hue_trung_binh2)
        
        cv.rectangle(anh_ket_qua2, (0, 0), (300, 70), (0, 0, 0), -1) 
        cv.putText(anh_ket_qua2, f"H_avg: {hue_trung_binh2:.1f}", (10, 25), font, 0.7, (0, 255, 255), 2)
        cv.putText(anh_ket_qua2, ket_luan2, (10, 55), font, 0.7, (0, 255, 0), 2)
        
        cv.imshow("Ket qua 2 (Medium)", anh_ket_qua2)

    # --- Xử lý hiển thị Ảnh 3 ---
    if anh_ket_qua3 is not None:
        ket_luan3 = phan_loai_xoai(hue_trung_binh3)
        
        cv.rectangle(anh_ket_qua3, (0, 0), (300, 70), (0, 0, 0), -1) 
        cv.putText(anh_ket_qua3, f"H_avg: {hue_trung_binh3:.1f}", (10, 25), font, 0.7, (0, 255, 255), 2)
        cv.putText(anh_ket_qua3, ket_luan3, (10, 55), font, 0.7, (0, 255, 0), 2)
        
        cv.imshow("Ket qua 3 (Xanh)", anh_ket_qua3)

    # --- BƯỚC 5: ĐỢI VÀ ĐÓNG ---
    cv.waitKey(0)
    cv.destroyAllWindows()