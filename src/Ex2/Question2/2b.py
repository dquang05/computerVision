import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]
    pick = []

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = areas[i] + areas[idxs[1:]] - intersection
        iou = intersection / (union + 1e-6)

        idxs = np.delete(idxs, np.concatenate(([0], np.where(iou > overlap_thresh)[0] + 1)))

    return boxes[pick]


# Get image path
base_dir = os.path.dirname(os.path.abspath(__file__))

# Concatenate full paths to image files
source_path = os.path.join(base_dir, "source_pcb.png")
template1_path = os.path.join(base_dir, "capacitor.png")
template2_path = os.path.join(base_dir, "res1.png")
template3_path = os.path.join(base_dir, "res2.png")

# Read source image
img_color = cv2.imread(source_path)
if img_color is None:
    raise FileNotFoundError(f"Cannot read source image: {source_path}")

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

templates = [
    {"path": template1_path, "threshold": 0.56, "color": (0, 255, 0)},
    {"path": template2_path, "threshold": 0.75, "color": (255, 0, 0)},
    {"path": template3_path, "threshold": 0.74, "color": (0, 0, 255)}
]

for t in templates:
    template = cv2.imread(t["path"], cv2.IMREAD_GRAYSCALE)

    if template is None:
        print(f"Không thể đọc được file: {t['path']}")
        continue

    h, w = template.shape

    # Use NCC TM_CCOEFF_NORMED to search for template (slightly different from 2a, using 2a would be very difficult)
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    locations = np.where(result >= t["threshold"])

    if locations[0].size == 0:
        print(f"Template {t['path']} not found with threshold = {t['threshold']}")
        continue

    boxes = []
    for pt in zip(*locations[::-1]):
        score = result[pt[1], pt[0]]
        boxes.append([pt[0], pt[1], w, h, score])

    print(f"Found {len(boxes)} rough positions for template {os.path.basename(t['path'])}")

    boxes_nms = non_max_suppression(boxes, overlap_thresh=0.3)

    for box in boxes_nms:
        x, y, w_box, h_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img_color, (x, y), (x + w_box, y + h_box), t["color"], 2)

    print(f"Kept {len(boxes_nms)} positions for template {os.path.basename(t['path'])}")

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Template Matching Recognition Result")
plt.axis("off")
plt.show()