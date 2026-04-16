import cv2 as cv
import numpy as np
import random

def phan_loai_xoai(h_value):

    if h_value < 52:
        return "CHIN (Ripe)"
    elif 52 <= h_value < 70:
        return "VUA (Medium)"
    else:
        return "SONG (Unripe)"

def sample_color_and_calculate_hue(img, mask, num_points=3):
    # Find all points in mango region
    points_mango = cv.findNonZero(mask)

    # Check if there are enough points for random sampling
    if points_mango is None or len(points_mango) < num_points:
        print("Error: Mango region not found or too small!")
        return None, None

    # Randomly select points
    random_indices = random.sample(range(len(points_mango)), num_points)
    
    display_img = img.copy()
    total_hue = 0 
    
    print(f"{'Point':<5} | {'R,G,B':<15} | {'Hue (Standard)':<15}")
    print("-" * 40)

    for i, idx in enumerate(random_indices):
        x, y = points_mango[idx][0]
        
        # 1. Get BGR color from original image
        b, g, r = img[y, x]
        
        # 2. Calculate Hue from RGB
        pixel_bgr = np.uint8([[[b, g, r]]]) 
        pixel_hsv = cv.cvtColor(pixel_bgr, cv.COLOR_BGR2HSV)
        
        # --- MODIFIED HERE: Multiply 2 to get 360-degree color space ---
        h_val = pixel_hsv[0][0][0] * 2  
        
        total_hue += h_val 
        
        # In thông tin
        print(f"#{i+1:<4} | {r},{g},{b:<11} | {h_val}")

        # Vẽ lên ảnh hiển thị
        cv.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv.putText(display_img, f"{r},{g},{b}", (x+10, y), 
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Calculate mean
    avg_hue = total_hue / num_points
    return avg_hue, display_img
    # 1. Read images
img_original1 = cv.imread('Chin.jpg', cv.IMREAD_COLOR) 
img_original2 = cv.imread('Medium.jpg', cv.IMREAD_COLOR) 
img_original3 = cv.imread('Xanh.jpg', cv.IMREAD_COLOR) 
img1 = cv.resize(img_original1, (500, 500))
img2 = cv.resize(img_original2, (500, 500))
img3 = cv.resize(img_original3, (500, 500))


    # --- STEP 1: BACKGROUND SEPARATION ---
hsv_img1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
hsv_img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
hsv_img3 = cv.cvtColor(img3, cv.COLOR_BGR2HSV)

    # Mask for blue background
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([140, 255, 255])
mask_background1 = cv.inRange(hsv_img1, lower_blue, upper_blue)
mask_background2 = cv.inRange(hsv_img2, lower_blue, upper_blue)
mask_background3 = cv.inRange(hsv_img3, lower_blue, upper_blue)

    # Invert mask to get mango
mask_mango1 = cv.bitwise_not(mask_background1)
mask_mango2 = cv.bitwise_not(mask_background2)
mask_mango3 = cv.bitwise_not(mask_background3)
kernel = np.ones((5, 5), np.uint8)
mask_mango1 = cv.morphologyEx(mask_mango1, cv.MORPH_OPEN, kernel)
mask_mango1 = cv.morphologyEx(mask_mango1, cv.MORPH_CLOSE, kernel)
mask_mango2 = cv.morphologyEx(mask_mango2, cv.MORPH_OPEN, kernel)
mask_mango2 = cv.morphologyEx(mask_mango2, cv.MORPH_CLOSE, kernel)
mask_mango3 = cv.morphologyEx(mask_mango3, cv.MORPH_OPEN, kernel)
mask_mango3 = cv.morphologyEx(mask_mango3, cv.MORPH_CLOSE, kernel)

    # --- STEP 2: PROCESS 3 POINTS ---
points_mango1 = cv.findNonZero(mask_mango1)
points_mango2 = cv.findNonZero(mask_mango2)
points_mango3 = cv.findNonZero(mask_mango3)


    # --- STEP 3: CALCULATION (Call function for each image) ---
hue_avg1, result_img1 = sample_color_and_calculate_hue(img1, mask_mango1, num_points=3)
hue_avg2, result_img2 = sample_color_and_calculate_hue(img2, mask_mango2, num_points=3)
hue_avg3, result_img3 = sample_color_and_calculate_hue(img3, mask_mango3, num_points=3)

    # --- STEP 4: DISPLAY RESULTS ---
font = cv.FONT_HERSHEY_SIMPLEX

    # --- Processing and displaying Image 1 ---
if result_img1 is not None:
        # 1. Classification
    conclusion1 = classify_mango(hue_avg1)
        
        # 2. Draw on result image 1
    cv.rectangle(result_img1, (0, 0), (300, 70), (0, 0, 0), -1) 
    cv.putText(result_img1, f"H_avg: {hue_avg1:.1f}", (10, 25), font, 0.7, (0, 255, 255), 2)
    cv.putText(result_img1, conclusion1, (10, 55), font, 0.7, (0, 255, 0), 2)
        
        # 3. Show window
    cv.imshow("Result 1 (Ripe)", result_img1)

    # --- Processing and displaying Image 2 ---
    if result_img2 is not None:
        conclusion2 = classify_mango(hue_avg2)
        
        cv.rectangle(result_img2, (0, 0), (300, 70), (0, 0, 0), -1) 
        cv.putText(result_img2, f"H_avg: {hue_avg2:.1f}", (10, 25), font, 0.7, (0, 255, 255), 2)
        cv.putText(result_img2, conclusion2, (10, 55), font, 0.7, (0, 255, 0), 2)
        
        cv.imshow("Result 2 (Medium)", result_img2)

    # --- Processing and displaying Image 3 ---
    if result_img3 is not None:
        conclusion3 = classify_mango(hue_avg3)
        
        cv.rectangle(result_img3, (0, 0), (300, 70), (0, 0, 0), -1) 
        cv.putText(result_img3, f"H_avg: {hue_avg3:.1f}", (10, 25), font, 0.7, (0, 255, 255), 2)
        cv.putText(result_img3, conclusion3, (10, 55), font, 0.7, (0, 255, 0), 2)
        
        cv.imshow("Result 3 (Unripe)", result_img3)

    # --- STEP 5: WAIT AND CLOSE ---
    cv.waitKey(0)
    cv.destroyAllWindows()