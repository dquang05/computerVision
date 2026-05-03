import cv2
import numpy as np
from pathlib import Path

# Use Path to handle relative directories correctly across different OS
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "image" / "502.jpg"

def create_calibration_data(image_path):
    """
    GUI to collect pixel coordinates by clicking on the image.
    """
    # Convert Path object to string for OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot find image at: {image_path}")
        return None

    screen_height = 800
    h, w = img.shape[:2]
    
    # Calculate scaling factor for display
    scaling_factor = screen_height / h if h > screen_height else 1.0
    display_img = cv2.resize(img, (int(w * scaling_factor), int(h * scaling_factor)), 
                             interpolation=cv2.INTER_AREA)

    points = []
    window_name = 'Calibration Image - Click Points'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    draw_img = display_img.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal draw_img
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map displayed coordinates back to original image scale
            orig_x = int(x / scaling_factor)
            orig_y = int(y / scaling_factor)
            points.append((orig_x, orig_y))
            
            print(f"Clicked point: ({orig_x}, {orig_y})")
            
            # Draw marker and index
            cv2.circle(draw_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(draw_img, f"P{len(points)}", (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow(window_name, draw_img)

    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click points on the image. Press 'q' to finish.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return np.array(points)

def estimate_camera_matrix_and_pose(data):
    """
    Direct Linear Transformation (DLT) using SVD and the QR-trick for decomposition.
    Data format: [u, v, X, Y, Z]
    """
    img_pts = data[:, :2]
    world_pts = data[:, 2:]
    n = len(img_pts)
    P_mat = np.zeros((2 * n, 12))
    
    for i in range(n):
        X, Y, Z = world_pts[i]
        u, v = img_pts[i]
        P_mat[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        P_mat[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
        
    # SVD decomposition
    _, _, V = np.linalg.svd(P_mat)
    M = V[-1, :].reshape(3, 4)
    
    if M[0, 0] > 0: 
        M = -M
        
    M_norm = M / M[2, 3]
    H = M_norm[:, 0:3]
    T_col = M_norm[:, 3]

    # QR decomposition trick to find K and Rotation (Ro)
    Q, R_qr = np.linalg.qr(np.linalg.inv(H))
    K = np.linalg.inv(R_qr)
    Ro = Q.T
    
    # Adjust signs to ensure positive diagonal in K
    D = np.diag(np.sign(np.diag(K)))
    K = (K @ D) / (K @ D)[2, 2]
    Ro = D @ Ro
    
    T_vec = np.linalg.inv(K) @ T_col
    
    return M_norm, K, np.column_stack((Ro, T_vec))

def calculate_world_coordinates(M, img_pts, z_known):
    """
    Calculates (X, Y) real-world coordinates from (u, v) pixels given a known Z.
    """
    world_coords = []
    for u, v in img_pts:
        # Solve the system of linear equations derived from the projection matrix
        A_sys = np.array([
            [M[0, 0] - u * M[2, 0], M[0, 1] - u * M[2, 1]],
            [M[1, 0] - v * M[2, 0], M[1, 1] - v * M[2, 1]]
        ])
        B_sys = np.array([
            u * M[2, 2] * z_known + u * M[2, 3] - M[0, 2] * z_known - M[0, 3],
            v * M[2, 2] * z_known + v * M[2, 3] - M[1, 2] * z_known - M[1, 3]
        ])
        res = np.linalg.solve(A_sys, B_sys)
        world_coords.append([np.ceil(res[0]), np.ceil(res[1]), z_known])
        
    return np.array(world_coords)

if __name__ == "__main__":
    # 1. Provide your training data [u, v, X, Y, Z]
    # Ensure you include points from different Z heights (502, 528, 538)
    training_data = np.array([
        [1924, 472, 137, 106, 502],
        [1814, 196, 117, 152, 502],
        [2289, 321, 200, 133, 502],
        [1930, 480, 137, 106, 528],
        [1820, 205, 117, 152, 528],
        [2295, 330, 200, 133, 528]
    ])
    
    # 2. Run Calibration
    M, K, Extrinsic = estimate_camera_matrix_and_pose(training_data)
    
    print("\n--- CALIBRATION RESULTS ---")
    print("Intrinsic Matrix K:\n", K)
    print("\nExtrinsic Matrix [R|T]:\n", Extrinsic)

    # 3. GUI Verification
    print(f"\nOpening GUI for verification. Target Z: 502mm")
    test_pts = create_calibration_data(IMAGE_PATH)
    
    if test_pts is not None:
        world_results = calculate_world_coordinates(M, test_pts, 502)
        print("\n--- VERIFICATION RESULTS ---")
        for i, pt in enumerate(world_results):
            print(f"P{i+1} Pixel: {test_pts[i]} -> World: ({int(pt[0])}, {int(pt[1])})")