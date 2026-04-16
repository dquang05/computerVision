import cv2
import numpy as np

# --- 1. CONFIGURATION ---
width, height = 800, 800
red_color = (0, 0, 255)
move_speed = 0.1
rotate_speed = 3.0
rect_size = (3.0 * 80, 1.5 * 80)  # (240, 120)

# ---- added: padding to prevent cutting when rotating ----
half_diag = int(np.ceil(np.hypot(rect_size[0] / 2, rect_size[1] / 2)))
PADDING = half_diag + 8   # add extra margin for aesthetics
SCALE = (width - 2 * PADDING) / 10.0  # new scale to replace fixed 80

def to_px_center(x, y, w, h):
    """Map 10x10 system -> pixel with padding to prevent overflow when rotating"""
    center_x = PADDING + (x + w/2) * SCALE
    center_y = height - (PADDING + (y + h/2) * SCALE)
    return (center_x, center_y)

c1 = to_px_center(1, 8, 3, 1.5)    # Top - Left
c2 = to_px_center(6, 8, 3, 1.5)    # Top - Right
c4 = to_px_center(6, 1, 3, 1.5)    # Bottom - Right
c5 = to_px_center(1, 1, 3, 1.5)    # Bottom - Left
c3 = to_px_center(6.75, 4, 1.5, 3) # Middle - Right
c6 = to_px_center(1.75, 4, 1.5, 3) # Middle - Left

def draw_rotated_step(center, angle, text):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    rotated_rect = (center, rect_size, angle)
    box = cv2.boxPoints(rotated_rect)                 # (4,2) float
    box = box.reshape((-1, 1, 2)).astype(np.int32)    # (4,1,2) int32  ✅ standard polylines format

    # DRAW BORDER ONLY (no fill) - using positional parameters
    cv2.polylines(frame, [box], True, red_color, 3, cv2.LINE_AA)

    cv2.putText(frame, text, (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Complete Cycle Animation", frame)
    return cv2.waitKey(10)


# --- 2. LOGIC ĐIỀU KHIỂN CHUYỂN ĐỘNG ---

def move_to(target, start_pos, current_angle, label):
    curr_pos = list(start_pos)
    while True:
        dist = np.linalg.norm(np.array(target) - np.array(curr_pos))
        if dist < move_speed * 80:
            return tuple(target)

        direction = (np.array(target) - np.array(curr_pos)) / dist
        curr_pos[0] += direction[0] * move_speed * 80
        curr_pos[1] += direction[1] * move_speed * 80

        if draw_rotated_step(tuple(curr_pos), current_angle, label) == ord('q'):
            exit()


def rotate_to(target_angle, center, start_angle, label):
    curr_angle = start_angle
    while curr_angle < target_angle:
        curr_angle += rotate_speed
        if curr_angle > target_angle: curr_angle = target_angle

        if draw_rotated_step(center, curr_angle, label) == ord('q'):
            exit()
    cv2.waitKey(300)  # Nghỉ ngắn sau khi xoay
    return curr_angle


# --- 3. MAIN PROGRAM (CYCLE) ---

current_p = c1
angle = 0.0

while True:  # Infinite loop to create repeating cycle
    # Stage 1: P1 -> P2
    current_p = move_to(c2, current_p, angle, "Move: P1 -> P2")

    # Stage 2: Rotate 90 degrees at P2
    angle = rotate_to(90, current_p, angle, "Rotate 90 deg at P2")

    # Stage 3: P2 -> P3 -> P4 (Move downward)
    current_p = move_to(c3, current_p, angle, "Move: P2 -> P3")
    current_p = move_to(c4, current_p, angle, "Move: P3 -> P4")

    # Stage 4: Add another 90 degrees (to 180) at P4
    angle = rotate_to(180, current_p, angle, "Rotate 90 deg at P4")

    # Stage 5: P4 -> P5 (Move left)
    current_p = move_to(c5, current_p, angle, "Move: P4 -> P5")

    # Stage 6: Add another 90 degrees (to 270) at P5
    angle = rotate_to(270, current_p, angle, "Rotate 90 deg at P5")

    # Stage 7: P5 -> P6 -> P1 (Return to original position)
    current_p = move_to(c6, current_p, angle, "Move: P5 -> P6")
    current_p = move_to(c1, current_p, angle, "Move: P6 -> Back to P1")
    angle = rotate_to(360, current_p, angle, "Rotate 90 deg at P1")

    # Stage 8: Reset angle to 0 for smooth cycle restart
    angle = 0  # Or keep rotating to 360

    print("Completed one cycle!")

cv2.destroyAllWindows()
