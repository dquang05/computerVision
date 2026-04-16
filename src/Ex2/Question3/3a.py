# star_morph_only_with_horizontal_v2.py
# Morphology-only: DIAGONAL + HORIZONTAL edges (2-end gating) + REMOVE inner strokes using largest CC.

import cv2 as cv
import numpy as np
import os

P = dict(
    # Binarize
    blur_k=5,

    # DIAGONAL: line-opening
    angle_step = 3,
    len_ratio = 0.18,        # SE length for diagonal edges
    thick = 3,               # kernel thickness & for redrawing
    diag_min_deg = 15,       # diagonal angle range
    diag_max_deg = 75,

    # Small closing for diagonals
    close_gap = 3,

    # HORIZONTAL: long opening + shape filtering + 2-end gating
    horiz_len_ratio = 0.25,
    min_horiz_len_ratio = 0.22,
    max_horiz_thick_ratio = 0.03,
    anchor_dilate_ratio = 0.02,

    # CC filtering
    keep_only_largest_cc = True,  # << turn on to remove inner strokes
    min_area_ratio = 0.12,        # if you want to keep >1 CC, use ratio based on the largest CC
)

# ---------- helper functions ----------

# Create disk-shaped SE with radius r
def diskSE(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1))

# Create line kernel of length L, thickness t, angle ang_deg (degrees)
def line_kernel(L: int, t: int, ang_deg: float) -> np.ndarray:
    L = max(5, int(L)); t = max(1, int(t))
    k = L + 2*t + 2
    ker = np.zeros((k, k), np.uint8)
    c = k // 2
    rad = np.deg2rad(ang_deg)
    dx, dy = np.cos(rad)*(L/2), np.sin(rad)*(L/2)
    x1, y1 = int(round(c - dx)), int(round(c - dy))
    x2, y2 = int(round(c + dx)), int(round(c + dy))
    cv.line(ker, (x1, y1), (x2, y2), 255, thickness=t)
    return (ker > 0).astype(np.uint8)

# Check if angle a is in range [amin, amax] (degrees), considering both directions
def in_diag(a: float, amin: float, amax: float) -> bool:
    a = abs(a)
    return (amin <= a <= amax) or (amin <= (180 - a) <= amax)

# Find left-right extreme endpoints of CC in the mask
def cc_extreme_endpoints(mask_cc: np.ndarray):
    contours, _ = cv.findContours(mask_cc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours: return None, None
    pts = contours[0].reshape(-1, 2)
    iL, iR = np.argmin(pts[:,0]), np.argmax(pts[:,0])
    return tuple(pts[iL]), tuple(pts[iR])

# Keep the largest CC or CCs with area >= ratio compared to the largest CC
def keep_largest_or_by_ratio(mask: np.ndarray, keep_only_largest=True, min_area_ratio=0.1):
    fg = (mask > 0).astype(np.uint8)
    # FIX ERROR 69: Add keyword connectivity=8
    num, labels, stats, _ = cv.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1: return mask
    areas = stats[1:, cv.CC_STAT_AREA]
    max_area = areas.max()
    out = np.zeros_like(fg)
    for i in range(1, num):
        if keep_only_largest and stats[i, cv.CC_STAT_AREA] != max_area:
            continue
        if not keep_only_largest and stats[i, cv.CC_STAT_AREA] < max_area * min_area_ratio:
            continue
        out[labels == i] = 255
    return out

# ---------- main ----------
def main(img_path: str):
    # Read grayscale image
    src = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if src is None:
        print(f"Cannot open image: {img_path}")
        return
    
    # Pre-processing: blur + Otsu thresholding 
    # FIX ERROR 91: Cast blur_k to int
    blur_k = int(P["blur_k"])
    blur = cv.GaussianBlur(src, (blur_k, blur_k), 0.0)
    _, bin_inv = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # Image dimensions
    h, w = bin_inv.shape[:2]
    base = min(h, w)

    # 1) FIND AND FILTER DIAGONAL EDGES
    L_open = max(25, int(P["len_ratio"] * base))
    accum_diag = np.zeros_like(bin_inv)
    
    # FIX ERROR 101: Cast angle_step to int in range function
    for ang in range(0, 180, int(P["angle_step"])):
        if not in_diag(ang, float(P["diag_min_deg"]), float(P["diag_max_deg"])):
            continue
        # FIX ERROR 104: Cast thick to int
        ker = line_kernel(L_open, int(P["thick"]), ang)
        opened = cv.morphologyEx(bin_inv, cv.MORPH_OPEN, ker)
        accum_diag = cv.bitwise_or(accum_diag, opened)
        
    # FIX ERROR 107: Cast close_gap to int
    diag_mask = cv.morphologyEx(accum_diag, cv.MORPH_CLOSE, diskSE(int(P["close_gap"])))

    # Anchor from diagonals (slight dilation) to check horizontal endpoints
    anchor_r = max(2, int(P["anchor_dilate_ratio"] * base))
    diag_anchor = cv.dilate(diag_mask, diskSE(anchor_r), iterations=1)

    # 2) FIND AND FILTER HORIZONTAL EDGES
    Lh = max(30, int(P["horiz_len_ratio"] * base))
    horiz_open = cv.morphologyEx(bin_inv, cv.MORPH_OPEN,
                                 cv.getStructuringElement(cv.MORPH_RECT, (Lh, 1)))

    # Shape filtering + 2-end gating
    min_w = max(25, int(P["min_horiz_len_ratio"] * base))
    max_th = max(2,  int(P["max_horiz_thick_ratio"] * base))
    fgH = (horiz_open > 0).astype(np.uint8)
    
    # FIX ERROR 122: Add keyword connectivity=8
    num, labels, stats, _ = cv.connectedComponentsWithStats(fgH, connectivity=8)
    horiz_keep = np.zeros_like(fgH)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if bw < min_w or bh > max_th:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        pL, pR = cc_extreme_endpoints(comp)
        if pL is None: 
            continue
        def touches_anchor(p):
            x0 = max(0, p[0]-anchor_r); x1 = min(w, p[0]+anchor_r+1)
            y0 = max(0, p[1]-anchor_r); y1 = min(h, p[1]+anchor_r+1)
            return cv.countNonZero(diag_anchor[y0:y1, x0:x1]) > 0
        if touches_anchor(pL) and touches_anchor(pR):
            horiz_keep = cv.bitwise_or(horiz_keep, comp)

    # 3) MERGE + small closing
    merged = cv.bitwise_or(diag_mask, horiz_keep)
    
    # FIX ERROR 141: Cast close_gap to int
    clean = cv.morphologyEx(merged, cv.MORPH_CLOSE, diskSE(int(P["close_gap"])))

    # 4) **REMOVE inner strokes**: keep largest CC (star outline)
    # FIX ERROR 146: Explicitly cast bool and float
    clean = keep_largest_or_by_ratio(
        clean,
        keep_only_largest=bool(P["keep_only_largest_cc"]),
        min_area_ratio=float(P["min_area_ratio"])
    )

    result = cv.bitwise_not(clean)

    # Display
    cv.imshow("Input", src)
    cv.imshow("Diagonal mask", diag_mask)
    cv.imwrite("diagonal_mask.png", diag_mask)
    cv.imshow("Accepted horizontals", horiz_keep)
    cv.imwrite("accepted_horizontals.png", horiz_keep)
    cv.imshow("Merged before CC filter", merged)
    cv.imwrite("merged_before_cc_filter.png", merged)
    cv.imshow("Result", result)
    cv.imwrite("star_result.png", result)
    cv.waitKey(0); cv.destroyAllWindows()

if __name__ == "__main__":
    # find PNG files in the same directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    png_files = [f for f in os.listdir(cur_dir) if f.lower().endswith(".png")]

    if len(png_files) == 0:
        print("No PNG files found in the directory!")
    else:
        img_path = os.path.join(cur_dir, "ex3.png")
        print(f"Processing file: {img_path}")
        main(img_path)

# Execution order includes the following steps:
# 1. Read image, thresholding, then filter diagonal edges to obtain the "Diagonal mask" image.
# 2. Continue to filter horizontal edges to obtain the "Accepted horizontals" image.
# 3. Merge the 2 images before CC filtering to obtain the "Merged before CC filter" image.
# 4. Finally, perform CC filtering to remove inner redundant strokes of the star to obtain the "Result" image.