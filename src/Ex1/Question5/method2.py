import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv.imread('Xanh.jpg', cv.IMREAD_COLOR)

# Resize if needed (note: your original code declares resized but calculates hist on original img)
# If you want to calculate on small image for speed, change img variable below to resized
img = cv.resize(img, (250, 250))

color = ('b', 'g', 'r')
channel_names = ('Blue', 'Green', 'Red')

# Create a wide frame to contain 3 charts
plt.figure(figsize=(15, 5))

for i, col in enumerate(color):
    # Calculate histogram for channel i
    histr = cv.calcHist([img], [i], None, [256], [0, 256])

    # Create subplot: 1 row, 3 columns, position i+1
    plt.subplot(1, 3, i + 1)

    # Draw chart
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

    # Add title and grid for easy viewing
    plt.title(channel_names[i])
    plt.grid(True, alpha=0.3)
    plt.xlabel('Pixel Value')
    plt.ylabel('Quantity')

plt.tight_layout() # Auto adjust spacing
plt.show()