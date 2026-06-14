import numpy as np
import cv2

# =========================
# 1) Dữ liệu Ma trận (STT = 3)
# =========================

# Ma trận H1
H1 = np.array([
    [30, 5, 1, 3],
    [5, 15, 2, 4],
    [6, 8, 9, 5],
    [7, 10, 21, 18],
    [8, 9, 5, 10],
    [24, 33, 44, 55]
], dtype=np.float32)

# Ma trận H2
H2 = np.array([
    [30, 34, 11, 78],
    [8, 15, 15, 98],
    [9, 34, 9, 23],
    [10, 13, 21, 18],
    [11, 9, 89, 78],
    [24, 99, 11, 21]
], dtype=np.float32)

# Ma trận H3
H3 = np.array([
    [30, 8, 23, 0], # Thêm 0 để đủ 4 cột nếu ảnh bị thiếu
    [6, 15, 9, 56],
    [7, 34, 9, 76],
    [8, 56, 21, 18],
    [9, 9, 23, 23],
    [24, 33, 44, 55]
], dtype=np.float32)

# Ma trận Template
TEMPLATE = np.array([
    [30, 9, 7, 20],
    [5, 15, 7, 55],
    [7, 33, 9, 76],
    [8, 56, 21, 18],
    [8, 9, 22, 23],
    [24, 31, 43, 54]
], dtype=np.float32)

# =========================
# 2) Hàm PCA + Nhận dạng
# =========================
def as_row_sample(mat: np.ndarray) -> np.ndarray:
    # Flatten matrix (6x4) -> (1x24)
    return mat.reshape(1, -1).astype(np.float32)

def pca_train_and_recognize(train_mats, test_mat, k=2):
    # Stack training images into a single matrix X
    X = np.vstack([as_row_sample(m) for m in train_mats])
    x_test = as_row_sample(test_mat)

    # Perform PCA using OpenCV's PCACompute2
    # Returns mean, eigenvectors (rows), and eigenvalues
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(X, mean=None, maxComponents=k) # type: ignore

    # Project training data and template onto the PCA subspace
    proj_train = cv2.PCAProject(X, mean, eigenvectors)
    proj_test  = cv2.PCAProject(x_test, mean, eigenvectors)

    # Calculate Euclidean distances between template and training samples
    dists = np.linalg.norm(proj_train - proj_test, axis=1)
    best_idx = int(np.argmin(dists))

    return {
        "eigenvalues": eigenvalues,
        "proj_train": proj_train,
        "proj_test": proj_test,
        "dists": dists,
        "best_idx": best_idx
    }

# =========================
# 3) Thực thi và In kết quả
# =========================
labels = ["H1", "H2", "H3"]
train_set = [H1, H2, H3]

# Chạy thử nghiệm với k=2 (số thành phần chính)
for k in [2]:
    result = pca_train_and_recognize(train_set, TEMPLATE, k=k)
    
    print("Trị riêng:", result["eigenvalues"].ravel())
    
    print("\nKhoảng cách Euclid:")
    for i, d in enumerate(result["dists"]):
        print(f" - Đến {labels[i]}: {d:.6f}")
        
    print(f"\n=> Kết luận: Template giống với hình {labels[result['best_idx']]} nhất.")