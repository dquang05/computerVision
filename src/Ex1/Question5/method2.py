import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
img = cv.imread('Xanh.jpg', cv.IMREAD_COLOR)

# Resize nếu cần (lưu ý: code gốc của bạn khai báo resized nhưng tính hist trên img gốc)
# Nếu muốn tính trên ảnh nhỏ cho nhanh thì đổi biến img bên dưới thành resized
img = cv.resize(img, (250, 250))

color = ('b', 'g', 'r')
channel_names = ('Blue (Xanh dương)', 'Green (Xanh lá)', 'Red (Đỏ)')

# Tạo một khung hình rộng để chứa 3 biểu đồ
plt.figure(figsize=(15, 5))

for i, col in enumerate(color):
    # Tính histogram cho kênh màu thứ i
    histr = cv.calcHist([img], [i], None, [256], [0, 256])

    # Tạo ô biểu đồ con: 1 hàng, 3 cột, vị trí thứ i+1
    plt.subplot(1, 3, i + 1)

    # Vẽ biểu đồ
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

    # Thêm tiêu đề và lưới cho dễ nhìn
    plt.title(channel_names[i])
    plt.grid(True, alpha=0.3)
    plt.xlabel('Giá trị Pixel')
    plt.ylabel('Số lượng')

plt.tight_layout() # Tự động căn chỉnh khoảng cách
plt.show()