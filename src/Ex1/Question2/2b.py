import cv2
import numpy as np

def histogram(name: str, image: np.ndarray) -> None:
    # Number of bins and histogram height
    bins = 256
    hist_h = 255
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [bins], [0, 255])
    # Get max value for normalization
    max_val = float(hist.max()) if hist.max() > 0 else 1.0
    # Create histogram image
    hist_image = np.zeros((hist_h, bins, 3), dtype=np.uint8)
    # Draw histogram
    for i in range(bins):
        binV = float(hist[i][0])
        height = int(round(binV * hist_h / max_val))
        cv2.line(hist_image, (i, hist_h - height), (i, hist_h), (255, 255, 255), 1)

    cv2.imshow(name, hist_image)

def main():
    # Matrix from problem group 3
    anh = np.array([
        30, 20, 30, 30,
        30, 30, 100, 110,
        30, 30, 30, 150,
        30, 230, 240, 250
    ], dtype=np.float32)

    gray_anh = anh.reshape((4, 4))

    # Convert to uint8
    gray_anh_u8 = gray_anh.astype(np.uint8)

    # Equalize histogram
    gray_his = cv2.equalizeHist(gray_anh_u8)

    # Print matrix to console
    print("Matrix =\n", gray_anh_u8)
    print("\nMatrix_histogram =\n", gray_his)

    # Display to terminal
    scale = 100
    cv2.namedWindow("Gray_old", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Gray_new", cv2.WINDOW_FREERATIO)

    cv2.imshow("Gray_old", cv2.resize(gray_anh_u8, (4*scale, 4*scale), interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Gray_new", cv2.resize(gray_his, (4*scale, 4*scale), interpolation=cv2.INTER_NEAREST))

    # Display histogram for both images
    histogram("Histogram old", gray_anh_u8)
    histogram("Histogram New", gray_his)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
