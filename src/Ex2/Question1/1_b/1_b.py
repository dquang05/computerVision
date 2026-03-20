from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PATH_A = BASE_DIR / "A.png"
PATH_B = BASE_DIR / "B.png"

STT = 1
SIGMA = STT * 0.5

PATCH = 21
MAX_CORNERS = 120
NUM_POINTS = 5
MATCH_THRESHOLD = 0.65
MIN_DIST = 18


def preprocess(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (5, 5), SIGMA)


def front_roi(img):
    h, w = img.shape
    # Chỉ giữ vùng mặt trước, bỏ nền trên và cạnh phải
    x1, y1 = int(0.05 * w), int(0.25 * h)
    x2, y2 = int(0.78 * w), int(0.95 * h)
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def get_corners(img, mask):
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask,
        blockSize=3,
        useHarrisDetector=True,
        k=0.04
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2)


def get_patch(img, pt, size=PATCH):
    r = size // 2
    x, y = map(int, np.round(pt))
    if x - r < 0 or y - r < 0 or x + r >= img.shape[1] or y + r >= img.shape[0]:
        return None
    return img[y-r:y+r+1, x-r:x+r+1]


def score_patch(p1, p2):
    p1 = p1.astype(np.float32)
    p2 = p2.astype(np.float32)
    return float(cv2.matchTemplate(p2, p1, cv2.TM_CCOEFF_NORMED)[0, 0])


def best_match(pt, img_src, pts_dst, img_dst):
    patch_src = get_patch(img_src, pt)
    if patch_src is None:
        return None, -1

    best_pt = None
    best_score = -1

    for q in pts_dst:
        patch_dst = get_patch(img_dst, q)
        if patch_dst is None:
            continue

        s = score_patch(patch_src, patch_dst)
        if s > best_score:
            best_score = s
            best_pt = q

    return best_pt, best_score


def mutual_matches(cornersA, imgA, cornersB, imgB):
    matches = []

    for ptA in cornersA:
        ptB, sAB = best_match(ptA, imgA, cornersB, imgB)
        if ptB is None or sAB < MATCH_THRESHOLD:
            continue

        ptA_back, sBA = best_match(ptB, imgB, cornersA, imgA)
        if ptA_back is None:
            continue

        if np.linalg.norm(ptA - ptA_back) <= 3:
            matches.append({
                "ptA": tuple(map(int, np.round(ptA))),
                "ptB": tuple(map(int, np.round(ptB))),
                "score": float((sAB + sBA) / 2.0)
            })

    return matches


def geometric_filter(matches):
    if len(matches) < 3:
        return matches

    ptsA = np.array([m["ptA"] for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    ptsB = np.array([m["ptB"] for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        ptsA, ptsB,
        method=cv2.RANSAC,
        ransacReprojThreshold=8
    )

    if inliers is None:
        return matches

    filtered = [m for m, keep in zip(matches, inliers.ravel()) if keep]
    return filtered


def select_top(matches):
    matches = sorted(matches, key=lambda m: m["score"], reverse=True)
    final = []
    usedA, usedB = [], []

    for m in matches:
        pA = np.array(m["ptA"], dtype=np.float32)
        pB = np.array(m["ptB"], dtype=np.float32)

        okA = all(np.linalg.norm(pA - q) >= MIN_DIST for q in usedA)
        okB = all(np.linalg.norm(pB - q) >= MIN_DIST for q in usedB)

        if okA and okB:
            final.append(m)
            usedA.append(pA)
            usedB.append(pB)

        if len(final) == NUM_POINTS:
            break

    return final


def draw_result(imgA, imgB, matches):
    A = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
    B = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)

    h = max(A.shape[0], B.shape[0])
    wA, wB = A.shape[1], B.shape[1]

    canvas = np.zeros((h, wA + wB, 3), dtype=np.uint8)
    canvas[:A.shape[0], :wA] = A
    canvas[:B.shape[0], wA:wA+wB] = B

    for i, m in enumerate(matches, 1):
        xA, yA = m["ptA"]
        xB, yB = m["ptB"]

        pA = (xA, yA)
        pB = (xB + wA, yB)

        cv2.circle(canvas, pA, 8, (0, 0, 255), -1)
        cv2.circle(canvas, pB, 8, (0, 0, 255), -1)

        cv2.putText(canvas, str(i), (xA + 8, yA - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (xA + 8, yA - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(canvas, str(i), (xB + wA + 8, yB - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (xB + wA + 8, yB - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.line(canvas, pA, pB, (0, 255, 0), 2)

    return canvas


imgA = cv2.imread(PATH_A, cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread(PATH_B, cv2.IMREAD_GRAYSCALE)

if imgA is None or imgB is None:
    raise ValueError("Không đọc được ảnh A hoặc B")

A = preprocess(imgA)
B = preprocess(imgB)

maskA = front_roi(A)
maskB = front_roi(B)

cornersA = get_corners(A, maskA)
cornersB = get_corners(B, maskB)

matches = mutual_matches(cornersA, A, cornersB, B)
matches = geometric_filter(matches)
final_matches = select_top(matches)

for i, m in enumerate(final_matches, 1):
    print(f"Point {i}: A{m['ptA']} <-> B{m['ptB']}   score = {m['score']:.4f}")

result = draw_result(imgA, imgB, final_matches)
cv2.imwrite("corresponding_points_result_better.png", result)
cv2.namedWindow("Corresponding Points", cv2.WINDOW_NORMAL)

screen_w = 1400
if result.shape[1] > screen_w:
    scale = screen_w / result.shape[1]
    show = cv2.resize(result, None, fx=scale, fy=scale)
else:
    show = result

cv2.imshow("Corresponding Points", show)
cv2.waitKey(0)
cv2.destroyAllWindows()