import cv2
import numpy as np
import math
from pathlib import Path

# Cập nhật đường dẫn đến file ảnh (tương đối theo file này)
BASE_DIR = Path(__file__).resolve().parent
input_image_path = BASE_DIR / "img3.jpg"

def edge_filtering_pipeline(image_path):
    """
    Quy trình lọc cạnh đầy đủ: Đọc ảnh -> Xám -> Khử nhiễu -> Canny
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Lỗi: Không tìm thấy file ảnh tại đường dẫn: {image_path}")
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Lỗi: Không thể đọc được nội dung ảnh tại: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def keep_largest_outer_frame(edged_image):
    """
    Hàm đọc ảnh cạnh (output của Canny), giữ lại chỉ khung ngoài cùng bự nhất.
    """
    if edged_image is None:
        return None, None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Không tìm thấy đường bao nào.")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(edged_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=2)

    return largest_contour, mask

def find_contour_features_and_corners(largest_contour, image_to_draw_on):
    """
    Hàm tìm góc bằng xấp xỉ đa giác, đồng thời tính chu vi, diện tích, tâm và góc xoay.
    ĐẢM BẢO TRẢ VỀ ĐÚNG 4 GÓC.
    """
    peri = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)

    # Chiến lược tìm đúng 4 góc
    approx_corners = None

    # Cách 1: Tự động điều chỉnh epsilon để lấy chính xác 4 điểm
    for eps_mult in np.arange(0.01, 0.2, 0.005):
        epsilon = eps_mult * peri
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            approx_corners = approx
            break

    # Cách 2: Nếu không thể tìm được 4 góc (do vật thể bo tròn, nhiễu...) -> Dùng Box xấp xỉ
    if approx_corners is None:
        print("Cảnh báo: Không thể xấp xỉ chính xác 4 góc. Chuyển sang dùng Bounding Box!")
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box) # Ép kiểu về số nguyên
        approx_corners = box.reshape(4, 1, 2) # Đưa về format mảng 3 chiều giống approxPolyDP

    num_corners = len(approx_corners)

    # Tìm trọng tâm và góc xoay
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        mu20 = M['mu20'] / M['m00']
        mu02 = M['mu02'] / M['m00']
        mu11 = M['mu11'] / M['m00']
        theta_rad = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        angle_deg = math.degrees(theta_rad)
    else:
        cx, cy, angle_deg = 0, 0, 0

    print(f"\n--- KẾT QUẢ PHÂN TÍCH VẬT THỂ ---")
    print(f"Diện tích        : {area:.2f} pixel^2")
    print(f"Chu vi           : {peri:.2f} pixel")
    print(f"Trọng tâm (u, v) : ({cx}, {cy})")
    print(f"Góc xoay         : {angle_deg:.2f}°")
    print(f"Số lượng góc bắt được : {num_corners}")

    # Trích xuất tọa độ và vẽ lên ảnh
    corner_points = []

    # Sắp xếp lại góc (Tùy chọn nâng cao nếu cần góc vẽ theo thứ tự Trái-Trên, Phải-Trên...)
    # Ở đây ta giữ nguyên thứ tự Contour tìm được
    for i, point in enumerate(approx_corners):
        x, y = point[0]
        corner_points.append((x, y))
        print(f" -> Góc {i+1}: Tọa độ (u={x}, v={y})")

        cv2.circle(image_to_draw_on, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image_to_draw_on, str(i+1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.circle(image_to_draw_on, (cx, cy), 5, (255, 0, 0), -1)
    info_text = f"Angle: {angle_deg:.1f} deg"
    cv2.putText(image_to_draw_on, info_text, (cx - 50, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ĐÃ ĐÓNG PHẦN HIỂN THỊ CỦA HÀM NÀY
    cv2.namedWindow("Object Full Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Full Analysis", 1000, 600)
    cv2.imshow("Object Full Analysis", image_to_draw_on)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corner_points, area, peri, (cx, cy), angle_deg

def out_put(img_path, show=False):
    img_path = Path(img_path)
    edge_image = edge_filtering_pipeline(img_path)
    if edge_image is None:
        return []

    largest, _ = keep_largest_outer_frame(edge_image)
    if largest is None:
        return []

    draw_img = cv2.imread(str(img_path))
    corner_points, area, peri, center, angle = find_contour_features_and_corners(largest, draw_img)

    if show:
        cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Final Result", 800, 600)
        cv2.imshow("Final Result", draw_img) # type: ignore
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return corner_points

