import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
# -----------------------------
# General Configuration
# -----------------------------
img_resize_dim = (1000, 1000)
video_path = Path(__file__).resolve().parent / "starrunning.mp4"
print("Resolved video path:", video_path)

cap = cv.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Cannot open the video")
    exit()

# 1. Open video and get the first frame
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open the video")
    exit()

ret, frame_raw = cap.read()
cap.release()

if not ret:
    print("Cannot read the first frame")
    exit()

# -----------------------------
# Step 1. Preprocessing & Cropping Templates (Size Synchronization)
# -----------------------------
# Convert to grayscale BEFORE cropping
gray_img = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)

# Crop star corner regions from the image (Coordinates must match the current resolution)
# Note: Double check these coordinates. 
# If old coordinates (445:475, 580:610) belong to the original image, they will be incorrect on a resized one.
corner_img_1 = gray_img[445:470, 580:610].copy()
corner_img_2 = gray_img[445:470, 220:245].copy()
corner_img_3 = gray_img[320:345, 405:425].copy()
corner_img_4 = gray_img[630:653, 292:312].copy()
corner_img_5 = gray_img[630:653, 520:538].copy()

# Helper Functions
def rotate_kernel_int8(kernel, angle_deg):
    (h_k, w_k) = kernel.shape[:2]
    center = (w_k // 2, h_k // 2)
    rot_mat = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    # Use INTER_NEAREST to keep values -1, 0, 1 intact
    return cv.warpAffine(kernel, rot_mat, (w_k, h_k), flags=cv.INTER_NEAREST, borderValue=-1)

def rectanger(coords, img_result):
    for (y, x) in coords:
        # Draw a 20x20 square centered at (x, y)
        cv.rectangle(img_result, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 1)
        cv.circle(img_result, (x, y), 1, (0, 0, 255), -1)

def find_corner(corner_img, gray_img):
    if corner_img.size == 0:
        print("Cropped corner region is empty! Please check the crop coordinates.")
        exit()

    # Smooth to reduce video compression noise
    img_blurred = cv.GaussianBlur(gray_img, (5, 5), 0)
    corner_blurred = cv.GaussianBlur(corner_img, (3, 3), 0)

    # Convert to binary (0 or 255)
    _, binary_img = cv.threshold(img_blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    _, binary_corner = cv.threshold(corner_blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Convert to 0-1 format to create the kernel
    binary_corner_01 = binary_corner // 255

    # -----------------------------
    # Step 2. Create Flexible Hit-or-Miss Kernel
    # -----------------------------
    # Create a buffer region (Don't care) to increase matching capability
    k_se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    erode_k = cv.morphologyEx(binary_corner_01, cv.MORPH_ERODE, k_se, iterations=1)
    dilate_k = cv.morphologyEx(binary_corner_01, cv.MORPH_DILATE, k_se, iterations=2)
    dont_care = cv.subtract(dilate_k, erode_k)
    
    # Create int8 kernel: 1 (foreground), -1 (background), 0 (don't care)
    kernel_new = np.full(binary_corner_01.shape, -1, dtype=np.int8)
    kernel_new[erode_k == 1] = 1
    kernel_new[dont_care == 1] = 0
    
    # -----------------------------
    # Step 3. Hit-or-Miss Operation
    # -----------------------------
    final_output = np.zeros_like(binary_img)
    print("Scanning Hit-or-Miss...")
    hit_miss = cv.morphologyEx(binary_img, cv.MORPH_HITMISS, kernel_new)
    final_output = cv.bitwise_or(final_output, hit_miss)
    
    # -----------------------------
    # Step 4. Draw bounding boxes
    # -----------------------------
    # Find matching points (there might be multiple close points)
    coords = np.column_stack(np.where(final_output > 0))
    print(f"Number of matching points found: {len(coords)}")
    img_result = cv.cvtColor(binary_img, cv.COLOR_GRAY2BGR)
    rectanger(coords, img_result)
    
    return img_result, coords, kernel_new

# Execute corner detection
result_img1, detected_coords1, kernel_new1 = find_corner(corner_img_1, gray_img)
result_img2, detected_coords2, kernel_new2 = find_corner(corner_img_2, gray_img)
result_img3, detected_coords3, kernel_new3 = find_corner(corner_img_3, gray_img)
result_img4, detected_coords4, kernel_new4 = find_corner(corner_img_4, gray_img)
result_img5, detected_coords5, kernel_new5 = find_corner(corner_img_5, gray_img)

# Safely combine images to avoid uint8 overflow
final_img = result_img1.copy()
for img in [result_img2, result_img3, result_img4, result_img5]:
    final_img = cv.bitwise_or(final_img, img)

plt.figure(figsize=(10, 10))
# vmax changed to 255 because bitwise_or output ranges up to 255
plt.imshow(final_img, cmap='gray', vmin=0, vmax=255)
plt.title("Star Corner Detection Results")
plt.axis('off')
plt.show()


# ========================================================
# 1. AGGREGATE COORDINATES FROM THE 5 DETECTED CORNERS
# ========================================================
# Combine all coordinates from detected_coords1 -> 5 into a single list
all_detected_points = []

for coords in [detected_coords1, detected_coords2, detected_coords3, detected_coords4, detected_coords5]:
    for (y, x) in coords:
        # OpenCV requires (x, y) format while np.where returns (row, col) which is (y, x)
        all_detected_points.append([x, y])

# Convert to a float32 numpy array format for KLT
p0 = np.array(all_detected_points, dtype=np.float32).reshape(-1, 1, 2)

print(f"Total starting points for tracking: {len(p0)}")


# ========================================================
# 2. KLT OPTICAL FLOW AND VIDEO CONFIGURATION
# ========================================================
lk_params = dict(winSize=(20, 20),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

# Re-open the video for tracking
cap = cv.VideoCapture(video_path)

# Read the first frame for initialization
ret, first_frame = cap.read()
if not ret:
    print("Cannot read the video")
    cap.release()
    exit()

# Resize first frame to match the size of gray_img (where you found the coordinates)
# Important: Video size must match the image size used for finding corners
target_size = gray_img.shape[::-1] # (width, height)
old_frame_resized = cv.resize(first_frame, target_size)
old_gray = cv.cvtColor(old_frame_resized, cv.COLOR_BGR2GRAY)


# ========================================================
# 3. TRACKING LOOP
# ========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Preprocess current frame
    frame_resized = cv.resize(frame, target_size)
    frame_gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)

    # 2. Calculate Optical Flow (KLT)
    # p1 is new coordinates, st is status (1 if found, 0 if lost)
    if p0 is None or len(p0) == 0:
        print("No points to track")
        cap.release()
        cv.destroyAllWindows()
        exit()

    p1, st, err = cv.calcOpticalFlowPyrLK(
    old_gray,
    frame_gray,
    p0,
    None,
    winSize=(20, 20),
    maxLevel=5,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    # 3. Filter good tracking points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # 4. Draw results on the frame
        for i, new in enumerate(good_new):
            a, b = new.ravel().astype(int)
            
            # Draw center point (red)
            cv.circle(frame_resized, (a, b), 3, (0, 0, 255), -1)
            
            # Draw 20x20 square (a-10 to a+10) around the point (blue)
            cv.rectangle(frame_resized, (a - 10, b - 10), (a + 10, b + 10), (255, 0, 0), 1)

        # Update p0 for the next frame
        p0 = good_new.reshape(-1, 1, 2)
    
    # 5. Display
    cv.imshow("KLT Star Tracking - Pre-detected Points", frame_resized)

    # Save gray frame to compare with the next frame
    old_gray = frame_gray.copy()

    if cv.waitKey(30) & 0xFF == 27: # Press ESC to exit
        break

cap.release()
cv.destroyAllWindows()