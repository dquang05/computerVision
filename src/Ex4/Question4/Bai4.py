import cv2
import numpy as np
import math


def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự chuẩn: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect


def match_corners_with_previous(current, prev):
    """Khớp 4 góc của frame hiện tại theo frame trước để tránh lật hình khi xoay"""
    matched = np.zeros_like(prev)
    used = [False] * 4

    for i in range(4):
        min_dist = float("inf")
        min_idx = -1
        for j in range(4):
            if used[j]:
                continue
            dist = np.linalg.norm(prev[i] - current[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        matched[i] = current[min_idx]
        used[min_idx] = True

    return matched


def main():
    video_path = 'movie.mp4'
    image_path = 'image.png'

    cap = cv2.VideoCapture(video_path)
    source_image = cv2.imread(image_path)

    if not cap.isOpened() or source_image is None:
        print("Lỗi: Không thể đọc được video hoặc ảnh nguồn. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Thông số video và giả định Intrinsic Matrix (K)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    h_src, w_src = source_image.shape[:2]
    src_points = np.array([
        [0, 0],
        [w_src - 1, 0],
        [w_src - 1, h_src - 1],
        [0, h_src - 1]
    ], dtype="float32")

    K = np.array([
        [width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    # Hệ thống Tracking Data
    tracked_boxes = {}
    next_box_id = 0
    tracking_data = []
    frame_count = 0
    kernel = np.ones((5, 5), np.uint8)

    print("Đang xử lý video, vui lòng đợi...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 2. Tiền xử lý không gian màu và lọc hình thái học
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

        current_detections = []

        # 3. Trích xuất đa giác và tính toán trọng tâm bằng Image Moments
        for cnt in valid_contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) != 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                corners = np.int32(box).astype("float32")
            else:
                corners = approx.reshape(4, 2).astype("float32")

            # Sử dụng cv2.moments để tính tâm đa giác chính xác
            M_moment = cv2.moments(corners)
            if M_moment["m00"] != 0:
                cx = M_moment["m10"] / M_moment["m00"]  # m10 map to x (column)
                cy = M_moment["m01"] / M_moment["m00"]  # m01 map to y (row)
                center = np.array([cx, cy], dtype="float32")
            else:
                center = np.mean(corners, axis=0)

            current_detections.append({'center': center, 'corners': corners})

        # 4. Gắn nhãn ID và Khóa góc xoay (Multi-Target Tracking)
        updated_tracked_boxes = {}

        if len(tracked_boxes) == 0:
            current_detections = sorted(current_detections, key=lambda d: d['center'][0])
            for idx, det in enumerate(current_detections):
                ordered_corners = order_points(det['corners'])
                updated_tracked_boxes[idx] = {
                    'center': det['center'],
                    'corners': ordered_corners,
                    'prev_center': det['center']
                }
            next_box_id = len(current_detections)
        else:
            for det in current_detections:
                best_id = -1
                min_dist = float("inf")

                for box_id, box_data in tracked_boxes.items():
                    dist = np.linalg.norm(det['center'] - box_data['center'])
                    if dist < min_dist:
                        min_dist = dist
                        best_id = box_id

                if best_id != -1 and min_dist < 100:
                    matched_corners = match_corners_with_previous(det['corners'], tracked_boxes[best_id]['corners'])
                    updated_tracked_boxes[best_id] = {
                        'center': det['center'],
                        'corners': matched_corners,
                        'prev_center': tracked_boxes[best_id]['center']
                    }
                else:
                    ordered_corners = order_points(det['corners'])
                    updated_tracked_boxes[next_box_id] = {
                        'center': det['center'],
                        'corners': ordered_corners,
                        'prev_center': det['center']
                    }
                    next_box_id += 1

        tracked_boxes = updated_tracked_boxes

        # 5. Phân tách Homography, Warp Perspective và xuất dữ liệu
        for box_id, box_data in tracked_boxes.items():
            corners = box_data['corners']
            center = box_data['center']
            prev_center = box_data['prev_center']

            M, _ = cv2.findHomography(src_points, corners)

            translation, roll, pitch, yaw = 0.0, 0.0, 0.0, 0.0

            if M is not None:
                translation = np.linalg.norm(center - prev_center)

                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(M, K)
                if num > 0:
                    R = Rs[0]
                    pitch = math.atan2(R[2, 1], R[2, 2]) * 180 / math.pi
                    yaw = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)) * 180 / math.pi
                    roll = math.atan2(R[1, 0], R[0, 0]) * 180 / math.pi

                tracking_data.append(f"{frame_count},{box_id},{translation:.2f},{roll:.2f},{pitch:.2f},{yaw:.2f}\n")

                warped = cv2.warpPerspective(source_image, M, (width, height))
                mask_bg = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask_bg, corners.astype(int), 255)
                mask_bg_3ch = cv2.cvtColor(mask_bg, cv2.COLOR_GRAY2BGR)

                bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask_bg_3ch))
                fg = cv2.bitwise_and(warped, mask_bg_3ch)
                frame = cv2.add(bg, fg)

        cv2.imshow("Projected video", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # 6. Đóng gói và lưu kết quả
    with open("data.txt", "w") as f:
        f.write("Frame,BoxID,Translation,Roll,Pitch,Yaw\n")
        f.writelines(tracking_data)

    print("Hoàn tất! Dữ liệu Tracking đã được ghi thành công vào file 'data.txt'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()