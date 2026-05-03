import numpy as np
import math
from pathlib import Path
from livision import out_put

BASE_DIR = Path(__file__).resolve().parent
input_image_path = BASE_DIR / "img3.jpg"

# Khai báo trực tiếp ma trận 3x3 và vectơ
K_matrix = np.array([
    [3.84348896e+03, 0.00000000e+00, 1.00263233e+03],
    [0.00000000e+00, 3.84560121e+03, 1.19370403e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

R_matrix = np.array([
    [-0.99593118,  0.01201359,  0.08931275],
    [-0.02264815, -0.99263228, -0.1190303 ],
    [ 0.08722473, -0.12056876,  0.98886552]
])

tvec = np.array([
    [ 63.46819515],
    [121.7834737 ],
    [514.58434178]
])

# Hàm tính toán quy đổi sang milimet
def pixel_to_world_mm_direct(u, v, K, R_matrix, tvec):
    Rt = np.column_stack((R_matrix[:, 0], R_matrix[:, 1], tvec.flatten()))
    H = K.dot(Rt)
    H_inv = np.linalg.inv(H)
    p_uv = np.array([[u], [v], [1.0]])
    P_world = H_inv.dot(p_uv)
    s = P_world[2][0]
    X_w = P_world[0][0] / s
    Y_w = P_world[1][0] / s
    return X_w, Y_w

# Xử lý chính
world_points = []
# Gọi hàm out_put từ file livision.py
corner_points = out_put(input_image_path, show=False)

for i in range(len(corner_points)):
    u, v = corner_points[i]
    X, Y = pixel_to_world_mm_direct(u, v, K_matrix, R_matrix, tvec)
    world_points.append((X, Y))

if len(world_points) >= 4:
    # 1. In tọa độ 4 góc
    print("\nWorld coordinates (Xw, Yw, Zw) quy đổi từ 4 góc (mm):")
    for i in range(4):
        X, Y = world_points[i]
        print(f"Corner {i}: Xw = {X:.3f}, Yw = {Y:.3f}, Zw = -0.000")
    print()

    # Lấy tọa độ 4 điểm
    X1, Y1 = world_points[0]
    X2, Y2 = world_points[1]
    X3, Y3 = world_points[2]
    X4, Y4 = world_points[3]

    # 2. In khoảng cách theo đúng yêu cầu cũ
    distance_1_to_2 = math.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
    distance_1_to_3 = math.sqrt((X1 - X3)**2 + (Y1 - Y3)**2)
    distance_1_to_4 = math.sqrt((X1 - X4)**2 + (Y1 - Y4)**2)

    print(f"Khoảng cách từ Điểm 1 tới Điểm 2: {distance_1_to_2:.2f} mm")
    print(f"Khoảng cách từ Điểm 1 tới Điểm 3: {distance_1_to_3:.2f} mm")
    print(f"Khoảng cách từ Điểm 1 tới Điểm 4: {distance_1_to_4:.2f} mm")
    print()

    # 3. Tính nốt 2 cạnh còn lại để tìm Max/Min (Kích thước Card)
    edge_23 = math.sqrt((X2 - X3)**2 + (Y2 - Y3)**2)
    edge_34 = math.sqrt((X3 - X4)**2 + (Y3 - Y4)**2)
    
    # Gom 4 cạnh bao quanh tứ giác: 1->2, 2->3, 3->4, 4->1 (chính là distance_1_to_4)
    edges = [distance_1_to_2, edge_23, edge_34, distance_1_to_4]
    
    max_edge = max(edges)
    min_edge = min(edges)

    print(">> ƯỚC LƯỢNG KÍCH THƯỚC CARD (theo mặt phẳng bàn cờ):")
    print(f"Chiều dài (max cạnh): {max_edge:.3f} mm")
    print(f"Chiều rộng (min cạnh): {min_edge:.3f} mm")

else:
    print("Lỗi: Không tìm đủ 4 điểm góc để thực hiện phép tính khoảng cách!")