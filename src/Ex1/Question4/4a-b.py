import cv2
import numpy as np

#Danh sách bán kính
radious_list = [1,5,10,15,20,25,30,35,40]

#Kích thước ảnh sau khi resize
target_resize = (100,100)

#Danh sách ảnh
images = ["anh_1.jpg","anh_2.jpg","anh_3.jpg","anh_4.jpg"]

#Biến lưu số pixel và số pixel trắng trên mỗi bán kính
white_pixels = np.zeros((len(images),len(radious_list)),dtype=int)

#a) Đọc số pixel trắng trên mỗi bán kính và ghi ra file result.txt
with open('result.txt','w') as f:
    for i, image_path in enumerate (images):
        f.write(f"Image {i+1}: {image_path}\n")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #Load ảnh gray
        if img is None:
            print(f"Khong the load duoc {image_path}")
            continue
        #Resize lại ảnh
        img_resize = cv2.resize(img, target_resize)
        
        #Xác định tâm vòng tròn
        center_x, center_y = target_resize[0] // 2, target_resize[1] // 2
        
        for j, r in enumerate(radious_list):
            count = 0
            for theta in np.arange(0, 2 * np.pi, 0.05):
                x = int(center_x + r * np.cos(theta))
                y = int(center_y + r * np.sin(theta))

                #Kiểm tra ngưỡng: 128
                if (0 < x < target_resize[0] and 0 < y < target_resize[1]):
                    if img_resize[y, x] > 128:
                        count += 1
            white_pixels[i][j] = count
            f.write(f"Ban kinh: {r}, so pixel trang: {count}\n")
        f.write("\n")
        
# b) Compare to identify the chessman.
# Hàm Normalized Cross-Correlation (NCC)
def ncc(x, y):
    #Đưa dữ liệu vector thành mảng để tính toán.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Tính trung bình các vector đặc trưng của mỗi hình
    mx = np.mean(x)
    my = np.mean(y)

    # Tính tử số của công thức NCC
    num = np.sum((x - mx) * (y - my))
    
    # Tính mẫu số
    den = np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))

    return 0.0 if den == 0 else num / den

# Chọn hệ số tương quan r = 0.85 
threshold = 0.85

# So sánh từng cặp ảnh (1-2, 1-3, 1-4, 2-3, 2-4, 3-4)
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        R = ncc(white_pixels[i], white_pixels[j])
        
        print(f"Ti le giong nhau cua anh {i+1} anh {j+1}: {R:.3f}")
        if R > threshold:
            print(f"Hai anh {i+1} va {j+1} giong nhau\n")
        else:
            print(f"Hai anh {i+1} va {j+1} khong giong nhau\n")
