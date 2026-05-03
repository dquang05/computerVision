import cv2
import numpy as np
from pathlib import Path

# Resolve image path relative to this file
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "image" / "502.jpg"

def create_calibration_data(image_path):
    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        print("Không thể tải ảnh. Vui lòng kiểm tra lại đường dẫn.")
        return None

    # Tự động lấy kích thước màn hình hoặc cố định một kích thước hiển thị nhỏ hơn
    screen_height = 800
    height, width = img.shape[:2]
    
    # Tính tỷ lệ thu nhỏ (Scale down)
    if height > screen_height:
        scaling_factor = screen_height / height
        new_width = int(width * scaling_factor)
        new_height = screen_height
        display_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        scaling_factor = 1.0
        display_img = img.copy()

    points = []
    window_name = 'Calibration Image - Click Points'
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Khởi tạo một biến chứa ảnh để vẽ lên
    draw_img = display_img.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal draw_img
        if event == cv2.EVENT_LBUTTONDOWN:
            # Quy đổi tọa độ từ ảnh đang hiển thị về lại tọa độ của ảnh gốc
            orig_x = int(x / scaling_factor)
            orig_y = int(y / scaling_factor)
            
            points.append((orig_x, orig_y))
            print(f"Đã chọn điểm: ({orig_x}, {orig_y})")
            
            # Vẽ hình tròn và số thứ tự tại điểm vừa click
            cv2.circle(draw_img, (x, y), 5, (0, 255, 0), -1)
            text = f"P{len(points)}"
            cv2.putText(draw_img, text, (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Hiển thị ảnh đã cập nhật
            cv2.imshow(window_name, draw_img)

    # Hiển thị cửa sổ GUI
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Nhấp vào các điểm trên ảnh theo thứ tự. Nhấn phím 'q' để dừng và lưu dữ liệu.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if not points:
        print("Không có điểm nào được chọn.")
        return None

    return points

def estimate_camera_matrix_and_pose(data):
    """Tính toán ma trận chiếu M, ma trận nội K và ngoại tại Ext."""
    img_coor = data[:, :2]
    world_coor = data[:, 2:]
    num_points = len(img_coor)
    P = np.zeros((2 * num_points, 12))
    
    j = 0
    for i in range(0, 2 * num_points, 2):
        P[i, 0:3] = world_coor[j, 0:3]
        P[i+1, 4:7] = world_coor[j, 0:3]
        P[i, 3] = 1
        P[i+1, 7] = 1
        P[i, 8:12] = -np.concatenate([world_coor[j, 0:3], [1]]) * img_coor[j, 0]
        P[i+1, 8:12] = -np.concatenate([world_coor[j, 0:3], [1]]) * img_coor[j, 1]
        j += 1
        
    U, S, V = np.linalg.svd(P)
    M = V[-1, :].reshape(3, 4)
    
    if M[0, 0] > 0: M = -M
    M_norm = M / M[2, 3] 
    
    H = M_norm[:, 0:3]
    T_col = M_norm[:, 3]
    Q, R_qr = np.linalg.qr(np.linalg.inv(H))
    
    K = np.linalg.inv(R_qr)
    Ro = Q.T
    D = np.diag(np.sign(np.diag(K)))
    K = (K @ D) / (K @ D)[2, 2]
    Ro = D @ Ro
    
    T_vec = np.linalg.inv(K) @ T_col
    return M, K, np.column_stack((Ro, T_vec))

def calculate_world_coordinates(M, img_pts, z_known):
    world_coords = []
    for pt in img_pts:
        u, v = pt[0], pt[1]
        A = np.array([
            [M[0, 0] - u * M[2, 0], M[0, 1] - u * M[2, 1]],
            [M[1, 0] - v * M[2, 0], M[1, 1] - v * M[2, 1]]
        ])
        B = np.array([
            u * M[2, 2] * z_known + u * M[2, 3] - M[0, 2] * z_known - M[0, 3],
            v * M[2, 2] * z_known + v * M[2, 3] - M[1, 2] * z_known - M[1, 3]
        ])
        res = np.linalg.solve(A, B)
        world_coords.append([np.ceil(res[0]), np.ceil(res[1]), z_known])
    return np.array(world_coords)

if __name__ == "__main__":
    data = np.array([
        [ 811,  633, -50,  75, 502],
        [1293,  828,  30,  45, 502],
        [1711,  652, 100,  75, 502],
        [1696, 1456, 100, -60, 502],
        [ 583, 1620, -85, -85, 502],
        [ 775,  648, -50,  75, 528],
        [1284,  844,  30,  45, 528],
        [1725,  652, 100,  75, 528],
        [1718, 1509, 100, -60, 528],
        [ 542, 1677, -85, -85, 528],
        [ 758,  679, -50,  75, 538],
        [1281,  885,  30,  45, 538],
        [1735,  703, 100,  75, 538],
        [1720, 1574, 100, -60, 538],
        [ 513, 1735, -85, -85, 538]
    ], dtype=float)

    Projected_matrix, K, Ext = estimate_camera_matrix_and_pose(data)
    
    np.set_printoptions(precision=8, suppress=False, formatter={'float': '{: .8e}'.format})
    print("MA TRẬN NỘI K")
    print(K)
    print("\n MA TRẬN NGOẠI (R và T)")
    print(Ext)
    
    print("\n" + "="*80)
    print(f"Đang tải ảnh từ: {IMAGE_PATH}")
    img_coor_test = create_calibration_data(IMAGE_PATH)
    
    if img_coor_test:
       
        z_known = 502 
        img_coor_test = np.array(img_coor_test)
        
        world_coor_3D = calculate_world_coordinates(Projected_matrix, img_coor_test, z_known)
        
        for i in range(len(img_coor_test)):
            print(f"Img_coordinate at P{i+1}: ({img_coor_test[i][0]}, {img_coor_test[i][1]}) -> "
                  f"World_Coordinate: ({int(world_coor_3D[i][0])}, {int(world_coor_3D[i][1])})")
 
        # Tính khoảng P1 với P2
        p1 = world_coor_3D[0]
        p2 = world_coor_3D[1]
        p3 = world_coor_3D[2]
        distance1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        distance2 = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2 + (p3[2] - p1[2])**2)
        
        print(f"Khoảng cách giữa P1({int(p1[0])}, {int(p1[1])}) và P2({int(p2[0])}, {int(p2[1])}) "
                f"là: {distance1:.6f} mm")
        print(f"Khoảng cách giữa P1({int(p1[0])}, {int(p1[1])}) và P3({int(p3[0])}, {int(p3[1])}) "
                f"là: {distance2:.6f} mm")