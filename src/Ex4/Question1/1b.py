import cv2
import numpy as np
import os

def find_similar_images_pca(template_path, folder_path, top_n=3, n_components=50):
    # ==========================================
    # BƯỚC 1: Đọc và chuẩn bị dữ liệu
    # ==========================================
    irow, icol = 128, 128
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    # Kiểm tra xem folder có tồn tại không
    if not os.path.exists(folder_path):
        return [], None, None, None, [], f"Lỗi: Không tìm thấy thư mục '{folder_path}'"

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
            
    if len(image_files) == 0:
        return [], None, None, None, [], f"Lỗi: Không có ảnh nào trong '{folder_path}'"
    
    faces_list = []
    valid_files = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            continue   
        img_resized = cv2.resize(img, (icol, irow))
        img_flat = img_resized.flatten()
        faces_list.append(img_flat)
        valid_files.append(img_file)
        
    faces = np.array(faces_list, dtype=np.float64)
    
    # ==========================================
    # BƯỚC 2: Tính Mean Face
    # ==========================================
    m = np.mean(faces, axis=0)
    mean_face_img = m.reshape(irow, icol)
    
    # ==========================================
    # BƯỚC 3: Trừ Mean - Chuẩn hóa dữ liệu
    # ==========================================
    faces_mean = faces - m
    
    # ==========================================
    # BƯỚC 4: Tính Covariance Matrix rút gọn và Eigenvalues
    # ==========================================
    # Sử dụng trick: L = A * A^T
    L = np.dot(faces_mean, faces_mean.T)
    eigenvalues, V = np.linalg.eig(L)
    
    # Sắp xếp trị riêng giảm dần
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # ==========================================
    # BƯỚC 5: Tính Eigenfaces (Principal Components)
    # ==========================================
    # Tính Eigenfaces thực sự: PC = A^T * V
    PC = np.dot(faces_mean.T, V)
    
    # BỔ SUNG QUAN TRỌNG: Trực chuẩn hóa các Eigenfaces
    for i in range(PC.shape[1]):
        norm_val = np.linalg.norm(PC[:, i])
        if norm_val > 0:
            PC[:, i] = PC[:, i] / norm_val
            
    # Giữ lại n_components quan trọng nhất
    N = min(n_components, PC.shape[1])
    PC = PC[:, :N]
    
    # ==========================================
    # BƯỚC 6: Tạo Signatures cho Database
    # ==========================================
    signatures = np.dot(faces_mean, PC)
    
    # ==========================================
    # BƯỚC 7: Xử lý Template Image
    # ==========================================
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return [], None, None, None, [], f"Lỗi: Không thể đọc template tại '{template_path}'"
    
    recognize = cv2.resize(template, (icol, irow))
    rec = recognize.flatten() - m
    rec_weighted = np.dot(rec, PC)
    
    # ==========================================
    # BƯỚC 8: Tính Euclidean distance
    # ==========================================
    scores = np.zeros(len(valid_files))
    for i in range(len(valid_files)):
        scores[i] = np.linalg.norm(signatures[i, :] - rec_weighted)
    
    # ==========================================
    # BƯỚC 9: Ranking và Tìm Top N
    # ==========================================
    idx = np.argsort(scores)
    
    top_images = []
    for i in range(min(top_n, len(idx))):
        img_name = valid_files[idx[i]]
        distance = scores[idx[i]]
        top_images.append((img_name, distance))
    
    # Tổng hợp thông tin để in ra màn hình
    info = {
        'n_images': len(image_files),
        'n_loaded': faces.shape[0],
        'L_shape': L.shape,
        'PC_shape': PC.shape,
        'n_components': N,
        'signatures_shape': signatures.shape
    }
    
    return top_images, mean_face_img, PC, signatures, valid_files, info

def main():
    print("--- CHƯƠNG TRÌNH ĐANG KHỞI ĐỘNG ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    template_path = os.path.join(script_dir, "template.jpg")
    folder_path = os.path.join(script_dir, "BTCPA2026") 
    
    print(f"Đang tìm ảnh tại: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"[LỖI CỰC NẶNG]: Vẫn không thấy folder tại {folder_path}")
        print("Hãy đảm bảo folder BTCPA2026 nằm cùng chỗ với file 1b.py")
        return

    top_n = 3
    n_components = 50
    
    # Gọi hàm PCA
    results = find_similar_images_pca(template_path, folder_path, top_n=top_n, n_components=n_components)
    
    # Kiểm tra kết quả (Phần này giữ nguyên như cũ)
    if isinstance(results[5], str):
        print(f"\n[LỖI]: {results[5]}")
        return
        
    top_images, mean_face, eigenfaces, signatures, valid_files, info = results
    
    print("\n" + "=" * 50)
    print("KẾT QUẢ TÌM KIẾM BẰNG EIGENFACES (PCA)")
    print("=" * 50)
    print(f" - Tổng số ảnh load được: {info['n_loaded']}") #type: ignore
    print(f" - Số lượng Eigenfaces giữ lại: {info['n_components']}") #type: ignore
    print("\nTOP 3 ẢNH GIỐNG NHẤT:")
    for i, (img_name, distance) in enumerate(top_images, 1):
        print(f"{i}. {img_name:<20} - Khoảng cách: {distance:.4f}")
    
   
    display_height, display_width = 250, 250
    images_to_show = []
    
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        print(f"Không thể đọc ảnh {template_path} để hiển thị.")
        return
        
    temp_display = cv2.resize(template_img, (display_width, display_height))
    temp_display_bgr = cv2.cvtColor(temp_display, cv2.COLOR_GRAY2BGR)
    cv2.putText(temp_display_bgr, "TEMPLATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    images_to_show.append(temp_display_bgr)
    
   
    mean_display = cv2.resize(mean_face.astype(np.uint8), (display_width, display_height))  #type: ignore
    mean_display_bgr = cv2.cvtColor(mean_display, cv2.COLOR_GRAY2BGR)
    cv2.putText(mean_display_bgr, "MEAN FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    images_to_show.append(mean_display_bgr)
    
    for i, (img_name, distance) in enumerate(top_images, 1):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (display_width, display_height)) #type: ignore
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(img_bgr, f"Top {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        images_to_show.append(img_bgr)
    
  
    final_display = cv2.hconcat(images_to_show)
    cv2.imshow("Ket qua PCA - Nhom 3", final_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()