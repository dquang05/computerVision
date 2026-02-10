import cv2
import numpy as np
import matplotlib.pyplot as plt

# Nhập ma trận
img = np.array([
    [20, 22, 22, 35, 39, 126],
    [198, 20, 98, 20, 56, 97],
    [18, 22, 203, 99, 102, 101],
    [32, 201, 198, 98, 100, 201],
    [197, 199, 203, 101, 22, 198],
    [78, 200, 46, 250, 89, 129]
], dtype=np.uint8)

# Step 1: Gaussian noise filtering
filtered = cv2.GaussianBlur(img, (3, 3), sigmaX=1)
print(filtered)
# Step 2: Reshape for K-means
data = filtered.reshape((-1, 1))
data = np.float32(data)

# Step 3: K-means segmentation (k = 3)
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(
    data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)
segmented = centers[labels.flatten()]
segmented = segmented.reshape(img.shape)

# Step 4: Display results
plt.figure(figsize=(9,3))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Gaussian Filtered")
plt.imshow(filtered, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Segmented (k=3)")
plt.imshow(segmented, cmap='gray')
plt.axis('off')

plt.show()
