import cv2
import numpy as np

# --- 1. CẤU HÌNH ---
width, height = 800, 800
red_color = (0, 0, 255)
move_speed = 0.1
rotate_speed = 3.0
rect_size = (3.0 * 80, 1.5 * 80)  # (240, 120)

# ---- thêm: padding để khi xoay không bị cắt ----
half_diag = int(np.ceil(np.hypot(rect_size[0] / 2, rect_size[1] / 2)))
PADDING = half_diag + 8   # chừa thêm 1 chút cho đẹp
SCALE = (width - 2 * PADDING) / 10.0  # scale mới thay cho 80 cố định

def to_px_center(x, y, w, h):
    """Map hệ 10x10 -> pixel, có padding để xoay không tràn khung"""
    center_x = PADDING + (x + w/2) * SCALE
    center_y = height - (PADDING + (y + h/2) * SCALE)
    return (center_x, center_y)

c1 = to_px_center(1, 8, 3, 1.5)    # Trên - Trái
c2 = to_px_center(6, 8, 3, 1.5)    # Trên - Phải
c4 = to_px_center(6, 1, 3, 1.5)    # Dưới - Phải
c5 = to_px_center(1, 1, 3, 1.5)    # Dưới - Trái
c3 = to_px_center(6.75, 4, 1.5, 3) # Giữa - Phải
c6 = to_px_center(1.75, 4, 1.5, 3) # Giữa - Trái

def draw_rotated_step(center, angle, text):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    rotated_rect = (center, rect_size, angle)
    box = cv2.boxPoints(rotated_rect)                 # (4,2) float
    box = box.reshape((-1, 1, 2)).astype(np.int32)    # (4,1,2) int32  ✅ chuẩn polylines

    # CHỈ VẼ VIỀN (không fill) - dùng tham số dạng positional
    cv2.polylines(frame, [box], True, red_color, 3, cv2.LINE_AA)

    cv2.putText(frame, text, (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Complete Cycle Animation", frame)
    return cv2.waitKey(10)


# --- 2. LOGIC ĐIỀU KHIỂN CHUYỂN ĐỘNG ---

def move_to(target, start_pos, current_angle, label):
    curr_pos = list(start_pos)
    while True:
        dist = np.linalg.norm(np.array(target) - np.array(curr_pos))
        if dist < move_speed * 80:
            return tuple(target)

        direction = (np.array(target) - np.array(curr_pos)) / dist
        curr_pos[0] += direction[0] * move_speed * 80
        curr_pos[1] += direction[1] * move_speed * 80

        if draw_rotated_step(tuple(curr_pos), current_angle, label) == ord('q'):
            exit()


def rotate_to(target_angle, center, start_angle, label):
    curr_angle = start_angle
    while curr_angle < target_angle:
        curr_angle += rotate_speed
        if curr_angle > target_angle: curr_angle = target_angle

        if draw_rotated_step(center, curr_angle, label) == ord('q'):
            exit()
    cv2.waitKey(300)  # Nghỉ ngắn sau khi xoay
    return curr_angle


# --- 3. CHƯƠNG TRÌNH CHÍNH (CHU KỲ) ---

current_p = c1
angle = 0.0

while True:  # Vòng lặp vô hạn để tạo chu kỳ lặp đi lặp lại
    # Giai đoạn 1: P1 -> P2
    current_p = move_to(c2, current_p, angle, "Di chuyen: P1 -> P2")

    # Giai đoạn 2: Xoay 90 độ tại P2
    angle = rotate_to(90, current_p, angle, "Xoay 90 do tai P2")

    # Giai đoạn 3: P2 -> P3 -> P4 (Di chuyển dọc xuống)
    current_p = move_to(c3, current_p, angle, "Di chuyen: P2 -> P3")
    current_p = move_to(c4, current_p, angle, "Di chuyen: P3 -> P4")

    # Giai đoạn 4: Xoay thêm 90 độ (thành 180) tại P4
    angle = rotate_to(180, current_p, angle, "Xoay 90 do tai P4")

    # Giai đoạn 5: P4 -> P5 (Di chuyển sang trái)
    current_p = move_to(c5, current_p, angle, "Di chuyen: P4 -> P5")

    # Giai đoạn 6: Xoay thêm 90 độ (thành 270) tại P5
    angle = rotate_to(270, current_p, angle, "Xoay 90 do tai P5")

    # Giai đoạn 7: P5 -> P6 -> P1 (Về vị trí ban đầu)
    current_p = move_to(c6, current_p, angle, "Di chuyen: P5 -> P6")
    current_p = move_to(c1, current_p, angle, "Di chuyen: P6 -> Ve P1")
    angle = rotate_to(360, current_p, angle, "Xoay 90 do tai P1")

    # Giai đoạn 8: Reset góc về 0 để bắt đầu chu kỳ mới mượt mà
    angle = 0  # Hoặc xoay tiếp lên 360

    print("Hoan thanh mot chu ky!")

cv2.destroyAllWindows()
