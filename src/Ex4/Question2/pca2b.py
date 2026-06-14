import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def choose_k_by_threshold(cumulative, threshold=0.95):
    # Find the minimum k that satisfies the threshold
    k = int(np.searchsorted(cumulative, threshold)) + 1
    k = max(1, min(k, len(cumulative)))
    return k

def choose_k_elbow(cumulative):
    # Find the elbow point using the maximum distance to the line
    cum = np.asarray(cumulative, dtype=float)
    n = len(cum)
    if n <= 1: return 1
    
    x = np.arange(1, n + 1)
    x1, y1 = 1, cum[0]
    x2, y2 = n, cum[-1]
    
    denom = np.hypot(x2 - x1, y2 - y1)
    if denom == 0: return 1
    
    distances = np.abs((y2 - y1) * x - (x2 - x1) * cum + x2 * y1 - y2 * x1) / denom
    elbow_idx = np.argmax(distances)
    return elbow_idx + 1

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

# Helper: find image file by searching script directory and working directory parents
def find_image(filename):
    script_dir = Path(__file__).resolve().parent
    # Search script dir and its parents first, then cwd and its parents
    candidates = [script_dir] + list(script_dir.parents) + [Path.cwd()] + list(Path.cwd().parents)
    seen = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        p = d / filename
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Image not found: {filename}. Searched script directory and working directory parents.")

# 1. Load image in color mode (BGR)
image_path = find_image("building.jpg")
img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img_color is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

img_float = img_color.astype(np.float32)
h, w, c = img_color.shape
print(f"Image dimensions: {h} x {w} x {c}")

# Split the color image into B, G, R channels
channels = cv2.split(img_float)
max_components = min(h, w)

# 2. Pre-compute PCA for each channel to optimize performance
# We do this ONCE instead of inside the loop
pca_results = []
for ch in channels:
    mean_ch, eigvecs_ch = cv2.PCACompute(ch, mean=None, maxComponents=max_components) #type: ignore
    pca_results.append({'mean': mean_ch, 'eigvecs': eigvecs_ch, 'data': ch})

# 3. Calculate Explained Variance (Using the first channel or Grayscale equivalent)
# We use the Green channel (index 1) or Grayscale to estimate the optimal k
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
mean_gray, eigvecs_gray = cv2.PCACompute(img_gray, mean=None, maxComponents=max_components) #type: ignore
Z_gray = cv2.PCAProject(img_gray, mean_gray, eigvecs_gray)

variances = np.var(Z_gray, axis=0, ddof=1)
total_var = np.sum(np.var(img_gray, axis=0, ddof=1))

if total_var <= 0:
    explained = np.zeros_like(variances)
else:
    explained = variances / total_var
cumulative = np.cumsum(explained)

# Calculate optimal k values
k_thresh = choose_k_by_threshold(cumulative, 0.95)
k_elbow = choose_k_elbow(cumulative)
print(f"Optimal k by 95% Threshold: {k_thresh}")
print(f"Optimal k by Elbow Heuristic: {k_elbow}")

# 4. Iteratively calculate MSE for the color image
max_k_for_mse = min(max_components, 100) # Limit plotting to 100 for better visualization
mse_list = []
k_range = range(1, max_k_for_mse + 1)

print("Calculating MSE for different k values...")
for k in k_range:
    reconstructed_channels = []
    
    for pca in pca_results:
        mean_k = pca['mean']
        # Slice the pre-computed eigenvectors to get only the top k
        eigvecs_k = pca['eigvecs'][:k, :] 
        
        # Project and BackProject
        Z_k = cv2.PCAProject(pca['data'], mean_k, eigvecs_k)
        recon_ch = cv2.PCABackProject(Z_k, mean_k, eigvecs_k)
        reconstructed_channels.append(recon_ch)
    
    # Merge channels back to BGR
    recon_img_float = cv2.merge(reconstructed_channels)
    
    # Calculate Mean Squared Error across all 3 channels
    mse = np.mean((img_float - recon_img_float) ** 2)
    mse_list.append(mse)

# 5. Plot MSE curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, mse_list, 'b-', linewidth=1.5, label='MSE')
if k_thresh <= max_k_for_mse:
    plt.axvline(x=k_thresh, color='red', linestyle='--', label=f'Threshold 95% (k={k_thresh})')
if k_elbow <= max_k_for_mse:
    plt.axvline(x=k_elbow, color='green', linestyle=':', label=f'Elbow (k={k_elbow})') #type: ignore

plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs Number of PCA Components (Color Image)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('mse_error_plot.png', dpi=200)
print("Saved MSE plot as 'mse_error_plot.png'")

# 6. Reconstruct and Display Final Images
def get_final_reconstruction(k_val):
    rec_channels = []
    for pca in pca_results:
        eig_k = pca['eigvecs'][:k_val, :]
        z_k = cv2.PCAProject(pca['data'], pca['mean'], eig_k)
        rec_ch = cv2.PCABackProject(z_k, pca['mean'], eig_k)
        rec_channels.append(rec_ch)
    # Merge and clip values to valid 8-bit range [0, 255]
    merged = cv2.merge(rec_channels)
    return np.clip(merged, 0, 255).astype(np.uint8)

img_recon_thresh = get_final_reconstruction(k_thresh)
img_recon_elbow = get_final_reconstruction(k_elbow)

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_thresh_rgb = cv2.cvtColor(img_recon_thresh, cv2.COLOR_BGR2RGB)
img_elbow_rgb = cv2.cvtColor(img_recon_elbow, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_thresh_rgb)
plt.title(f"Reconstructed (k={k_thresh}, 95% Var)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_elbow_rgb)
plt.title(f"Reconstructed (k={k_elbow}, Elbow)")
plt.axis('off')

plt.tight_layout()
plt.savefig('color_reconstructed_images.png', dpi=200)
plt.show()

print("\n===== SUMMARY =====")
print(f"Minimum components (95% variance): {k_thresh}")
print(f"Minimum components (Elbow method): {k_elbow}")
print("Conclusion: The elbow method typically provides the lowest optimal k that maintains visual fidelity.")