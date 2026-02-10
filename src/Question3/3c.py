import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
IMAGE_PATH = "Tiger.jpg"
OUTPUT_PATH = "segmented_auto_no_shadow.png"
USE_AUTO_MAPPING = True
N_SEGMENTS = 800
K_CLUSTERS = 4
CLAHE = True

# ---------- pattern generators ----------
def diagonal_stripes(h, w, stripe_w=12, col1=(255,140,0), col2=(0,0,0)):
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    diag = ((X + Y) // stripe_w) % 2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[diag == 0] = col1
    canvas[diag == 1] = col2
    return canvas

def crosshatch(h, w, spacing=12, line_col=(0,100,0), bg_col=(200,255,200), thickness=2):
    canvas = np.full((h, w, 3), bg_col, dtype=np.uint8)
    for x in range(-h, w, spacing):
        cv2.line(canvas, (x, 0), (x + h, h), line_col, thickness)
    for x in range(0, w + h, spacing):
        cv2.line(canvas, (x, 0), (x - h, h), line_col, thickness)
    return canvas

def dotted(h, w, spacing=16, dot_col=(150,110,80), bg_col=(245,240,230), r=3):
    canvas = np.full((h, w, 3), bg_col, dtype=np.uint8)
    for y in range(spacing//2, h, spacing):
        for x in range(spacing//2, w, spacing):
            cv2.circle(canvas, (x, y), r, dot_col, -1)
    return canvas

def solid_blue(h, w, color=(30,144,255)):
    canvas = np.full((h, w, 3), color, dtype=np.uint8)
    return canvas

# ---------- helpers ----------
def clean_mask_bool(mask_bool, ksize=5):
    m = (mask_bool.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (m > 0)

def apply_pattern_rgb(base_bgr, mask_bool, pattern_rgb, alpha=1.0):
    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    pat = cv2.resize(pattern_rgb, (base_rgb.shape[1], base_rgb.shape[0]))
    out = base_rgb.copy()
    for c in range(3):
        out[:,:,c][mask_bool] = (alpha * pat[:,:,c][mask_bool] + (1-alpha) * out[:,:,c][mask_bool]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# ---------- segmentation ----------
def superpixel_kmeans(img_bgr, n_segments=700, k=4):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    segments = slic(img_rgb, n_segments=n_segments, compactness=10, start_label=0)
    lab = rgb2lab(img_rgb)
    sp_ids = np.unique(segments)
    feats = np.array([lab[segments == s].mean(axis=0) for s in sp_ids], dtype=np.float32)
    km = KMeans(n_clusters=k, random_state=0).fit(feats)
    labels = np.zeros(segments.shape, dtype=np.int32)
    for idx, s in enumerate(sp_ids):
        labels[segments == s] = int(km.labels_[idx])
    return labels, km.cluster_centers_

def compute_cluster_mean_hsv(img_bgr, labels, k):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    centers_hsv = []
    for i in range(k):
        mask = (labels == i)
        if mask.sum() == 0:
            centers_hsv.append((0,0,0))
            continue
        vals = img_hsv[mask]
        mean = vals.mean(axis=0).astype(int)
        centers_hsv.append(tuple(int(x) for x in mean))
    return centers_hsv

def auto_map_by_hue(centers_hsv):
    mapping = {}
    hues = np.array([c[0] for c in centers_hsv])
    sats = np.array([c[1] for c in centers_hsv])
    assigned = set()
    for i,(h,s) in enumerate(zip(hues,sats)):
        if 90 <= h <= 140 and s > 40:
            mapping[i] = 'water'; assigned.add(i); break
    for i,(h,s) in enumerate(zip(hues,sats)):
        if i in assigned: continue
        if 35 <= h <= 90 and s > 40:
            mapping[i] = 'grass'; assigned.add(i); break
    for i,(h,s) in enumerate(zip(hues,sats)):
        if i in assigned: continue
        if (h <= 25 or h >= 160) and s > 40:
            mapping[i] = 'tiger'; assigned.add(i); break
    for i in range(len(centers_hsv)):
        if i not in assigned:
            mapping[i] = 'ground'
    return mapping

# ---------- GrabCut refine ----------
def refine_with_grabcut(img_bgr, init_mask_bool, iter_count=7):
    h, w = init_mask_bool.shape
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[init_mask_bool] = cv2.GC_PR_FGD
    sure_fg = cv2.erode(init_mask_bool.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=2).astype(bool)
    gc_mask[sure_fg] = cv2.GC_FGD
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)
    res_mask = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    return res_mask

# ---------- remove shadows under feet (focused) ----------
def remove_shadows_under_feet(img_bgr, tiger_mask_bool, v_thresh=70, y_expand=10, down_shift=30, min_area=200):
    h, w = tiger_mask_bool.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    ys, xs = np.where(tiger_mask_bool)
    if len(xs) == 0:
        return tiger_mask_bool
    x0, x1 = max(xs.min() - 5, 0), min(xs.max() + 5, w - 1)
    y0, y1 = max(ys.min() - y_expand, 0), min(ys.max() + y_expand, h - 1)
    y0_shift = min(h - 1, y0 + down_shift)
    y1_shift = min(h - 1, y1 + down_shift)
    if y0_shift > y1_shift:
        y0_shift = max(y0, 0)
        y1_shift = min(y1 + down_shift, h - 1)
    roi_v = V[y0_shift:y1_shift+1, x0:x1+1]
    shadow_roi = (roi_v < v_thresh)
    shadow_mask = np.zeros_like(tiger_mask_bool)
    shadow_mask[y0_shift:y1_shift+1, x0:x1+1] = shadow_roi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    dilated = cv2.dilate(tiger_mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    candidate = shadow_mask & dilated
    candidate_uint = (candidate.astype(np.uint8) * 255)
    num_labels, labels_im = cv2.connectedComponents(candidate_uint)
    cleaned = np.zeros_like(candidate)
    for lab in range(1, num_labels):
        comp = (labels_im == lab)
        if comp.sum() >= min_area:
            cleaned |= comp
    new_mask = tiger_mask_bool & (~cleaned)
    sure_fg = cv2.erode(new_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1) > 0
    new_mask = cv2.dilate(sure_fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2) > 0
    new_mask = clean_mask_bool(new_mask, ksize=5)
    return new_mask

# ---------- explicit blue detection (river) ----------
def detect_blue_regions(img_bgr, h_low=80, h_high=160, s_min=30, v_min=30, min_comp_area=500):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]; s = hsv[:,:,1]; v = hsv[:,:,2]
    blue_mask = ((h >= h_low) & (h <= h_high) & (s >= s_min) & (v >= v_min))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    blue_mask = cv2.morphologyEx((blue_mask.astype(np.uint8)*255), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    num_labels, labels_im = cv2.connectedComponents((blue_mask.astype(np.uint8)*255))
    cleaned = np.zeros_like(blue_mask)
    for lab in range(1, num_labels):
        comp = (labels_im == lab)
        if comp.sum() >= min_comp_area:
            cleaned |= comp
    return cleaned

# ---------- main ----------
def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("cv2.imread returned None. Check path/encoding.")

    img_proc = apply_clahe_bgr(img) if CLAHE else img.copy()
    labels, centers = superpixel_kmeans(img_proc, n_segments=N_SEGMENTS, k=K_CLUSTERS)
    centers_hsv = compute_cluster_mean_hsv(img, labels, K_CLUSTERS)
    mapping = auto_map_by_hue(centers_hsv) if USE_AUTO_MAPPING else {i: 'ground' for i in range(K_CLUSTERS)}

    h, w = labels.shape
    mask_tiger = np.zeros((h,w), dtype=bool)
    mask_grass = np.zeros((h,w), dtype=bool)
    mask_water = np.zeros((h,w), dtype=bool)
    mask_ground = np.zeros((h,w), dtype=bool)
    for i, role in mapping.items():
        if role == 'tiger':
            mask_tiger |= (labels == i)
        elif role == 'grass':
            mask_grass |= (labels == i)
        elif role == 'water':
            mask_water |= (labels == i)
        else:
            mask_ground |= (labels == i)

    mask_tiger = clean_mask_bool(mask_tiger, ksize=5)
    mask_grass = clean_mask_bool(mask_grass, ksize=5)
    mask_water = clean_mask_bool(mask_water, ksize=5)
    mask_ground = clean_mask_bool(mask_ground, ksize=5)

    blue_mask = detect_blue_regions(img)
    if blue_mask.sum() > 1000:
        mask_water = blue_mask
        mask_grass &= ~mask_water
        mask_ground &= ~mask_water
        mask_tiger &= ~mask_water

    refined = refine_with_grabcut(img, mask_tiger, iter_count=8)
    refined = clean_mask_bool(refined, ksize=5)

    # remove shadows under feet (focused)
    refined_no_shadow = remove_shadows_under_feet(img, refined,
                                                  v_thresh=70,
                                                  y_expand=8,
                                                  down_shift=30,
                                                  min_area=150)
    refined_no_shadow = clean_mask_bool(refined_no_shadow, ksize=5)

    mask_grass &= ~refined_no_shadow
    mask_ground &= ~refined_no_shadow
    mask_water &= ~refined_no_shadow

    mask_grass = clean_mask_bool(mask_grass, ksize=5)
    mask_ground = clean_mask_bool(mask_ground, ksize=5)
    mask_water = clean_mask_bool(mask_water, ksize=5)

    stripe = diagonal_stripes(h, w, stripe_w=12, col1=(255,140,0), col2=(0,0,0))
    hatch = crosshatch(h, w, spacing=12, line_col=(0,100,0), bg_col=(200,255,200), thickness=2)
    dots = dotted(h, w, spacing=16, dot_col=(150,110,80), bg_col=(245,240,230), r=3)
    blue = solid_blue(h, w, color=(30,144,255))

    base = cv2.GaussianBlur(img, (5,5), 0)
    out = base.copy()

    out = apply_pattern_rgb(out, mask_water, blue, alpha=1.0)
    out = apply_pattern_rgb(out, mask_ground, dots, alpha=1.0)
    out = apply_pattern_rgb(out, mask_grass, hatch, alpha=1.0)
    out = apply_pattern_rgb(out, refined_no_shadow, stripe, alpha=0.95)

    cv2.imwrite(OUTPUT_PATH, out)
    cv2.imwrite("debug_tiger_mask.png", (refined_no_shadow.astype(np.uint8)*255))
    cv2.imwrite("debug_water_mask.png", (mask_water.astype(np.uint8)*255))

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); plt.title("Result"); plt.axis('off')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()