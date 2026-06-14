import cv2
import numpy as np

# Biến toàn cục lưu 4 điểm click chuột
src_pts = []

def mouse_handler(event, x, y, flags, param):
    """Hàm callback bắt sự kiện click chọn góc sân"""
    global src_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(src_pts) < 4:
        src_pts.append((x, y))
        print(f"Point {len(src_pts)} selected: ({x}, {y})")

def main():
    global src_pts
    
    # PHẦN 1: CALIBRATION VÀ PERSPECTIVE TRANSFORMATION
    video_path = r"H:\Vision\3b\istockphoto-1399169622-640_adpp_is.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Lỗi: Không thể mở video!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 33 if fps <= 0 else int(round(1000 / fps))

    ret, frame = cap.read()
    if not ret:
        return

    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", mouse_handler)

    while True:
        temp_frame = frame.copy()
        for pt in src_pts:
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
            
        if len(src_pts) == 4:
            break
            
        cv2.imshow("Select 4 Corners", temp_frame)
        if cv2.waitKey(10) == 27: 
            return
            
    cv2.destroyWindow("Select 4 Corners")

    width, height = 300, 650
    dst_pts = np.array([
        [0, 0], 
        [width, 0], 
        [width, height], 
        [0, height]
    ], dtype=np.float32)
    
    src_pts_np = np.array(src_pts, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts_np, dst_pts)


    # PHẦN 2: BACKGROUND SUBTRACTION VÀ MORPHOLOGY
    pBackSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        fg_mask = pBackSub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Xóa nhiễu bằng Morphology
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        birdview = cv2.warpPerspective(frame, matrix, (width, height))

        # PHẦN 3: XỬ LÝ CONTOUR VÀ LOẠI TRỪ NGƯỜI CHƠI
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        transformed_contours = []
        player_exclusion_zones = []

        # Map contours sang tọa độ Bird-eye view  
        for c in contours:
            if len(c) > 0:
         
                c_float = c.astype(np.float32)
                c_transformed = cv2.perspectiveTransform(c_float, matrix)
                transformed_contours.append(c_transformed)
                
                area = cv2.contourArea(c_transformed)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(c_transformed)
                    pad = 20
                    player_exclusion_zones.append((x - pad, y - pad, x + w + pad, y + h + pad))

        # PHẦN 4: NHẬN DIỆN BÓNG (DỰA VÀO DIỆN TÍCH + VÙNG LOẠI TRỪ)
        if frame_count > 90:
            for fc in transformed_contours:
                area = cv2.contourArea(fc)

                # Lọc theo kích thước trên Bird-eye view
                if 5 <= area <= 180:
                    M = cv2.moments(fc)
                    if M["m00"] == 0:
                        continue
                    
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Bỏ qua nếu vật thể nằm quá gần người chơi (nhiễu từ vợt/tay/chân)
                    is_player_noise = False
                    for (x1, y1, x2, y2) in player_exclusion_zones:
                        if x1 <= cX <= x2 and y1 <= cY <= y2:
                            is_player_noise = True
                            break

                    if not is_player_noise:
                        # Vẽ điểm đỏ đặc cho quả bóng
                        cv2.circle(birdview, (cX, cY), 5, (0, 0, 255), -1)

        cv2.imshow("Original Video", frame)
        cv2.imshow("Bird-eye Ball Detection", birdview)
        if cv2.waitKey(delay) == 27:
            break
    cap.release()
    cv2.waitKey(1) 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()