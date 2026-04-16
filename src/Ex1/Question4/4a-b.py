import cv2
import numpy as np

# Radius list
radious_list = [1,5,10,15,20,25,30,35,40]

# Image size after resize
target_resize = (100,100)

# Image list
images = ["anh_1.jpg","anh_2.jpg","anh_3.jpg","anh_4.jpg"]

# Variable to store pixel count and white pixels per radius
white_pixels = np.zeros((len(images),len(radious_list)),dtype=int)

# a) Read white pixels per radius and write to result.txt
with open('result.txt','w') as f:
    for i, image_path in enumerate (images):
        f.write(f"Image {i+1}: {image_path}\n")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load grayscale image
        if img is None:
            print(f"Cannot load {image_path}")
            continue
        # Resize image
        img_resize = cv2.resize(img, target_resize)
        
        # Determine circle center
        center_x, center_y = target_resize[0] // 2, target_resize[1] // 2
        
        for j, r in enumerate(radious_list):
            count = 0
            for theta in np.arange(0, 2 * np.pi, 0.05):
                x = int(center_x + r * np.cos(theta))
                y = int(center_y + r * np.sin(theta))

                # Check threshold: 128
                if (0 < x < target_resize[0] and 0 < y < target_resize[1]):
                    if img_resize[y, x] > 128:
                        count += 1
            white_pixels[i][j] = count
            f.write(f"Radius: {r}, white pixels: {count}\n")
        f.write("\n")
        
# b) Compare to identify the chessman.
# Normalized Cross-Correlation (NCC) function
def ncc(x, y):
    # Convert vector data to array for calculation
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Calculate mean of feature vectors for each shape
    mx = np.mean(x)
    my = np.mean(y)

    # Calculate numerator of NCC formula
    num = np.sum((x - mx) * (y - my))
    
    # Calculate denominator
    den = np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))

    return 0.0 if den == 0 else num / den

# Choose correlation coefficient r = 0.85
threshold = 0.85

# Compare each image pair (1-2, 1-3, 1-4, 2-3, 2-4, 3-4)
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        R = ncc(white_pixels[i], white_pixels[j])
        
        print(f"Similarity ratio of image {i+1} and {j+1}: {R:.3f}")
        if R > threshold:
            print(f"Images {i+1} and {j+1} are similar\n")
        else:
            print(f"Images {i+1} and {j+1} are not similar\n")
