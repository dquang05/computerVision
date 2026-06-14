import numpy as np
import cv2
import matplotlib.pyplot as plt   # Thêm để vẽ ảnh

def print_for_word(title, matrix, is_integer=False, precision=3):
    """In ma trận không dấu ngoặc, căn lề để dán vào Word"""
    print(f"\n{title}")
    if matrix.ndim == 1:
        if is_integer:
            print("  ".join([f"{int(x):7d}" for x in matrix]))
        else:
            print("  ".join([f"{x:9.{precision}f}" for x in matrix]))
        return
    for row in matrix:
        if is_integer:
            print("  ".join([f"{int(x):7d}" for x in row]))
        else:
            print("  ".join([f"{x:9.{precision}f}" for x in row]))

# === Bước 1: Ma trận ảnh gốc (giống như trước) ===
A = np.array([
    [0,   0,   0, 255,   0,   0,   0],
    [0,   0, 255, 255, 255,   0,   0],
    [0, 255, 255, 255, 255, 255,   0],
    [0, 255,   0,   0,   0, 255,   0],
    [0, 255, 255, 255, 255, 255,   0],
    [0,   0, 255, 255, 255, 255,   0],
    [0,   0,   0, 255,   0,   0,   0]
], dtype=np.float32)   # OpenCV yêu cầu float32

print_for_word("Bước 1: Ma trận gốc A (OpenCV)", A, is_integer=True)

# === Bước 2: PCACompute (tính mean và eigenvectors) ===
num_components = 3   # chọn k=3
mean, eigenvectors = cv2.PCACompute(A, mean=None, maxComponents=num_components)#type: ignore

# mean là vector hàng, eigenvectors mỗi hàng là 1 vector riêng (đã sắp xếp giảm dần)
print_for_word("Bước 2 (OpenCV): Vector kỳ vọng (mean)", mean.flatten(), precision=2)
print_for_word("Bước 2 (OpenCV): Các vector riêng (mỗi hàng là 1 eigenvector)", eigenvectors, precision=3)

# === Bước 3: Chiếu dữ liệu (PCAProject) ===
Z = cv2.PCAProject(A, mean, eigenvectors)   # Z có shape (7, num_components)
print_for_word(f"Bước 3 (OpenCV): Dữ liệu giảm chiều Z (k={num_components})", Z, precision=3)

# === Bước 4: Phục hồi ảnh (PCABackProject) ===
A_rec = cv2.PCABackProject(Z, mean, eigenvectors)
print_for_word(f"Bước 4 (OpenCV): Ma trận khôi phục (k={num_components})", A_rec, precision=3)

# === Bước 5: Nhị phân hóa với ngưỡng 150 ===
A_bin = np.where(A_rec > 150, 255, 0).astype(np.uint8)
print_for_word(f"Bước 5 (OpenCV): Ảnh nhị phân (threshold=150, k={num_components})", A_bin, is_integer=True)

# === Làm lại với k = 1 (chỉ 1 thành phần chính) ===
print("\n" + "="*70)
print("TRƯỜNG HỢP k=1 (chỉ dùng 1 eigenvalue lớn nhất) - OpenCV")
print("="*70)

num_components_1 = 1
mean1, eig1 = cv2.PCACompute(A, mean=None, maxComponents=num_components_1)#type: ignore
Z1 = cv2.PCAProject(A, mean1, eig1)
A_rec1 = cv2.PCABackProject(Z1, mean1, eig1)
A_bin1 = np.where(A_rec1 > 150, 255, 0).astype(np.uint8)

print_for_word("Bước 2 (k=1): Vector kỳ vọng", mean1.flatten(), precision=2)
print_for_word("Bước 2 (k=1): Vector riêng thứ nhất", eig1, precision=3)
print_for_word("Bước 3 (k=1): Dữ liệu giảm chiều Z", Z1, precision=3)
print_for_word("Bước 4 (k=1): Ma trận khôi phục", A_rec1, precision=3)
print_for_word("Bước 5 (k=1): Ảnh nhị phân (threshold=150)", A_bin1, is_integer=True)

# === In trị riêng lớn nhất dạng nhị phân ===
X_centered = A - mean
S = (1/7) * np.dot(X_centered.T, X_centered)
evals_np, _ = np.linalg.eig(S)
evals_np = np.sort(evals_np)[::-1]
lambda1 = evals_np[0]
print(f"\nTrị riêng lớn nhất (λ1) = {lambda1:.6f}")
print(f"Biểu diễn nhị phân của λ1 (phần nguyên): {bin(int(lambda1))}")

# ========== PHẦN THÊM: HIỂN THỊ ẢNH TRỰC QUAN ==========
print("\n" + "="*70)
print("HIỂN THỊ ẢNH TRỰC QUAN (Ảnh gốc, tái tạo k=3, k=1)")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Ảnh gốc", "Tái tạo k=3 (nhị phân)", "Tái tạo k=1 (nhị phân)"]
images = [A.astype(np.uint8), A_bin, A_bin1]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    # In giá trị số lên từng ô
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(img[i, j]), ha='center', va='center',
                    color='red' if img[i, j] == 0 else 'black', fontsize=8)
plt.tight_layout()
plt.savefig("pca_comparison.png", dpi=150)
plt.show()

print("\nĐã lưu ảnh so sánh vào file 'pca_comparison.png'")

print("\n--- HOÀN THÀNH (OpenCV) ---")