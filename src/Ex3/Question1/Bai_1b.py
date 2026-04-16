import cv2
import numpy as np
import math
from pathlib import Path

# 1. Khởi tạo Video
# Lấy đường dẫn tương đối từ thư mục chứa script
script_dir = Path(__file__).parent
video_path = script_dir / 'video.mp4'

# Kiểm tra file có tồn tại không
if not video_path.exists():
    print(f"Lỗi: Không tìm thấy file video tại {video_path}")
    exit()

cap = cv2.VideoCapture(str(video_path))

# Tham số cấu hình
BUFFER_SIZE = 10
frame_buffer = []
lane_3_count = 0
counting_line_y = 500

# Giới hạn lane 3
pts_lane3 = np.array([
    [610, 350], [650, 350],
    [830, 520], [660, 520]
], np.int32).reshape((-1, 1, 2))
x_left_limit = int(610 + (660 - 610) * (counting_line_y - 350) / (520 - 350))
x_right_limit = int(650 + (830 - 650) * (counting_line_y - 350) / (520 - 350))

# Bộ nhớ Tracking (id: [last_x, last_y, counted_flag])
trackers = {}
next_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

# lấy 10 frame từ video để làm nền
    if len(frame_buffer) < BUFFER_SIZE:
        frame_buffer.append(gray)
        continue

# Lấy frame 11 trừ Trung bình 10 frame trước
    avg_bg = np.mean(frame_buffer, axis=0).astype(np.uint8)
    diff = cv2.absdiff(gray, avg_bg)

    # Morphology
    _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # Nối xe đứt đoạn và tách xe dính nhau
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   Kẻ đường trên video
    # 1. Vẽ đường Lane 3
    cv2.line(frame, (x_left_limit, counting_line_y), (x_right_limit, counting_line_y), (0, 255, 255), 3)
    text = "LANE 3"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    center_x = (x_left_limit + x_right_limit) // 2
    text_x = center_x - text_w // 2
    cv2.putText(frame, text, (text_x, counting_line_y - 15), font, font_scale, (0, 255, 255), thickness)

    current_frame_blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 800 < area < 25000:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy_bottom = x + w // 2, y + h

# Kiểm tra điểm tiếp xúc mặt đất của xe có trong đa giác Lane 3 không
            if cv2.pointPolygonTest(pts_lane3, (cx, cy_bottom), False) >= 0:
                current_frame_blobs.append((cx, cy_bottom, x, y, w, h))

    # --- LOGIC TRACKING & IN TERMINAL ---
    new_trackers = {}
    for (cx, cy, x, y, w, h) in current_frame_blobs:
        found_id = None
        for tid, tdata in trackers.items():
            # Tính khoảng cách Euclide để nhận diện xe từ frame trước
            dist = math.hypot(cx - tdata[0], cy - tdata[1])
            if dist < 60:
                found_id = tid
                break

        if found_id is not None:
            last_y = trackers[found_id][1]
            is_counted = trackers[found_id][2]

    # Nếu bánh xe vượt qua vạch ngang
            if not is_counted and last_y < counting_line_y <= cy:
                lane_3_count += 1
                is_counted = True
                print(f"New transport detected! Total: {lane_3_count}")

            new_trackers[found_id] = [cx, cy, is_counted]
            color = (0, 0, 255) if is_counted else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
            # Gán ID mới cho xe mới xuất hiện
            new_trackers[next_id] = [cx, cy, False]
            next_id += 1

    trackers = new_trackers
    frame_buffer.pop(0)
    frame_buffer.append(gray)

# Hiển thị tổng số xe lên video
    cv2.putText(frame, f"Vehicles count on Lane 3: {lane_3_count}", (30, 60), font, 1.2, (0, 255, 0), 3)

    cv2.imshow("Lane 3 Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print(f"Number of transports on Lane 3: {lane_3_count}")