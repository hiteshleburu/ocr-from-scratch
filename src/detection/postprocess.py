import numpy as np
from scipy import ndimage


def mask_to_boxes(mask, min_area=10):
    """
    mask: 2D numpy array, binary (0/1)
    Returns a list of (x1, y1, x2, y2) boxes, one per connected blob of 1s.
    """
    labeled_array, n_components = ndimage.label(mask)
    boxes = []

    for component_id in range(1, n_components + 1):
        ys, xs = np.where(labeled_array == component_id)
        if len(xs) < min_area:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        boxes.append((x1, y1, x2, y2))

    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def evaluate_boxes(pred_boxes, true_boxes, iou_threshold=0.5):
    matched_true = set()
    tp = 0

    for pb in pred_boxes:
        best_iou, best_idx = 0, -1
        for i, tb in enumerate(true_boxes):
            if i in matched_true:
                continue
            score = iou(pb, tb)
            if score > best_iou:
                best_iou, best_idx = score, i
        if best_iou >= iou_threshold:
            tp += 1
            matched_true.add(best_idx)

    fp = len(pred_boxes) - tp
    fn = len(true_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1