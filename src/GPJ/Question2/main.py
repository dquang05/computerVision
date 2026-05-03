import cv2
import numpy as np
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "contour.jpg"

CANNY_T1 = 50
CANNY_T2 = 150
KERNEL_SZ = 3

MAX_SHOW_WIDTH = 1000
RESULT_SCALE = 0.4


def show_resized(window_name, image, max_width=1000):
    # Resize image only for display
    height, width = image.shape[:2]

    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_show = cv2.resize(image, (new_width, new_height))
    else:
        image_show = image

    cv2.imshow(window_name, image_show)


def save_resized(output_path, image, scale=0.4):
    # Save resized image for report
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    cv2.imwrite(str(output_path), resized)


def main():
    # Read image
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    img_draw = img.copy()

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold with Otsu
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological filtering
    kernel = np.ones((KERNEL_SZ, KERNEL_SZ), np.uint8)
    thresh_open = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    thresh_clean = cv2.morphologyEx(
        thresh_open,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Edge image for checking
    edges = cv2.Canny(thresh_clean, CANNY_T1, CANNY_T2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No contour found.")
        return

    # Select the largest contour
    largest_cnt = max(contours, key=cv2.contourArea)

    # Compute center by moments
    M = cv2.moments(largest_cnt)
    if M["m00"] == 0:
        print("Invalid contour moment.")
        return

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Rotated bounding box
    rect = cv2.minAreaRect(largest_cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Draw rotated box
    box_contour = box.reshape((-1, 1, 2))
    cv2.drawContours(img_draw, [box_contour], -1, (0, 255, 0), 3)

    # Find the longest edge
    max_len = -1.0
    dx_best = 0.0
    dy_best = 0.0

    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]

        x1 = int(p1[0])
        y1 = int(p1[1])
        x2 = int(p2[0])
        y2 = int(p2[1])

        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)

        if length > max_len:
            max_len = length
            dx_best = dx
            dy_best = dy

    if max_len == 0:
        print("Invalid edge length.")
        return

    # Compute angle from the longest edge
    angle_rad = math.atan2(dy_best, dx_best)
    angle_deg = math.degrees(angle_rad)

    # Normalize angle to (-90, 90]
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg <= -90:
        angle_deg += 180

    # Draw center
    cv2.circle(img_draw, (cx, cy), 8, (255, 0, 0), -1)

    # Draw direction line
    line_len = 150
    x_end = int(cx + line_len * dx_best / max_len)
    y_end = int(cy + line_len * dy_best / max_len)
    cv2.line(img_draw, (cx, cy), (x_end, y_end), (0, 255, 255), 3)

    # Draw result text
    text = f"Center: ({cx}, {cy})  Angle: {angle_deg:.2f} deg"
    cv2.putText(
        img_draw,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    # Print result
    print(f"Image path: {IMAGE_PATH}")
    print(f"Center: ({cx}, {cy})")
    print(f"Angle: {angle_deg:.2f} deg")

    # Save full-size images
    result_path = BASE_DIR / "result_contour.png"
    threshold_path = BASE_DIR / "threshold_contour.png"
    edges_path = BASE_DIR / "edges_contour.png"

    cv2.imwrite(str(result_path), img_draw)
    cv2.imwrite(str(threshold_path), thresh_clean)
    cv2.imwrite(str(edges_path), edges)

    # Save resized images for report
    save_resized(BASE_DIR / "result_contour_resized.png", img_draw, RESULT_SCALE)
    save_resized(BASE_DIR / "threshold_contour_resized.png", thresh_clean, RESULT_SCALE)
    save_resized(BASE_DIR / "edges_contour_resized.png", edges, RESULT_SCALE)

    print(f"Saved: {result_path}")
    print(f"Saved: {BASE_DIR / 'result_contour_resized.png'}")

    # Show resized windows
    show_resized("Threshold", thresh_clean, MAX_SHOW_WIDTH)
    show_resized("Edges", edges, MAX_SHOW_WIDTH)
    show_resized("Result", img_draw, MAX_SHOW_WIDTH)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()