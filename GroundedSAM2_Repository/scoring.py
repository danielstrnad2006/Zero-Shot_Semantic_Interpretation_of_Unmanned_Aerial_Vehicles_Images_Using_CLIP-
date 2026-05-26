# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np

# =============================================================================
# Process Masks and Labels
# =============================================================================
def init_prediction_state(h, w, empty_idx):
    pred_flat = np.full(h * w, empty_idx, dtype=int)
    min_areas_flat = np.full(h * w, np.inf)
    return pred_flat, min_areas_flat

def update_prediction_state(pred_flat, min_areas_flat, masks_batch, box_labels_batch, categories):
    valid_categories = [c for c in categories.keys() if c not in ['moving car', 'empty']]
    if masks_batch is None or len(masks_batch) == 0:
        return pred_flat, min_areas_flat
    masks_flat = masks_batch.reshape(masks_batch.shape[0], -1)
    num_masks = masks_flat.shape[0]
    mask_areas = masks_flat.sum(axis=1).astype(np.int32)
    predicted_indices = np.full(num_masks, categories['empty'][0], dtype=int)
    for i in range(min(num_masks, len(box_labels_batch))):
        phrase = str(box_labels_batch[i]).lower()
        for cat in valid_categories:
            if cat in phrase:
                predicted_indices[i] = categories[cat][0]
                break
    for i in range(num_masks):
        active_pixels = np.nonzero(masks_flat[i])[0]
        if active_pixels.size == 0:
            continue
        area_i = mask_areas[i]
        update_mask = area_i < min_areas_flat[active_pixels]
        pixels_to_update = active_pixels[update_mask]
        if pixels_to_update.size > 0:
            pred_flat[pixels_to_update] = predicted_indices[i]
            min_areas_flat[pixels_to_update] = area_i
    return pred_flat, min_areas_flat

def find_gt_mask(label, categories, category, h, w):
    def _color_array(color):
        return np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    if category == 'car':
        target_colors = [_color_array(categories['car'][1]), _color_array(categories['moving car'][1])]
    else:
        target_colors = [_color_array(categories[category][1])]
    gt_mask = np.zeros((h, w), dtype=bool)
    for color in target_colors:
        gt_mask |= np.all(label==color, axis=2)
    return gt_mask

def find_pred_pos(categories, category, pred_map):
    target_idx = categories[category][0]
    pred_pos = (pred_map == target_idx)
    if category == 'car':
        pred_pos = np.logical_or(pred_pos, pred_map == categories['moving car'][0])
    return pred_pos

# =============================================================================
# Compute Metrics
# =============================================================================
def calc_confusion_matrix(label, categories, category, h, w, pred_map):
    actual = find_gt_mask(label, categories, category, h, w)
    predicted = find_pred_pos(categories, category, pred_map)
    AP = int(actual.sum())
    TP = int(np.logical_and( predicted,  actual).sum())
    FP = int(np.logical_and( predicted, ~actual).sum())
    FN = int(np.logical_and(~predicted,  actual).sum())
    TN = int(np.logical_and(~predicted, ~actual).sum())
    return AP, TP, FP, TN, FN

def calc_precision(TP, FP):
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0.0
    return precision

def calc_recall(TP, FN):
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0.0
    return recall

def calc_Fscore(beta, precision, recall):
    try:
        Fscore = ((1 + (beta ** 2)) * precision * recall) / (((beta ** 2) * precision) + recall)
    except ZeroDivisionError:
        Fscore = 0.0
    return Fscore

def calc_IoU(TP, FP, FN):
    try:
        IoU = TP / (TP + FP + FN)
    except ZeroDivisionError:
        IoU = 0.0
    return IoU

def calc_mIoU(metrics):
    total_pixels = float(sum(metrics[c]['Actual Positives'] for c in metrics)) if metrics else 0.0
    IoUs = []
    weighted = []
    weight_sum = 0.0
    for category in metrics:
        IoU = metrics[category]['IoU']
        actual_pixels = float(metrics[category]['Actual Positives'])
        weight = actual_pixels / total_pixels if total_pixels > 0.0 else 0.0
        IoUs.append(IoU)
        weighted.append(IoU * weight)
        weight_sum += weight
    mIoU = np.mean(IoUs) if IoUs else 0.0
    FWIoU = np.sum(weighted) / weight_sum if weight_sum > 0.0 else 0.0
    return mIoU, FWIoU

def calc_metrics(label, pred_map, categories):
    h, w = label.shape[:2]
    metrics = {}
    for c in categories:
        if c not in ['moving car', 'empty']:
            AP, TP, FP, TN, FN = calc_confusion_matrix(label, categories, c, h, w, pred_map)
            precision = calc_precision(TP, FP)
            recall = calc_recall(TP, FN)
            Fscore = calc_Fscore(categories[c][2], precision, recall)
            IoU = calc_IoU(TP, FP, FN)
            metrics[c] = {
                'Actual Positives': AP,
                'True Positives': TP,
                'False Positives': FP,
                'True Negatives': TN,
                'False Negatives': FN,
                'Precision': precision,
                'Recall': recall,
                'F-Score': Fscore, 
                'IoU': IoU
            }
    mIoU, FWIoU = calc_mIoU(metrics)
    return metrics, mIoU, FWIoU

# =============================================================================
# Save Metrics to Excel
# =============================================================================
def save_metrics(file_id, excel, metrics, mIoU, FWIoU, excel_category_order):
    row = file_id+5
    column = 0
    excel.range((row, 1)).value = file_id+1
    excel.range((row, 2)).value = mIoU
    excel.range((row, 3)).value = FWIoU
    for c in excel_category_order:
        if c in metrics:
            excel.range((row, column+5 )).value = metrics[c]['IoU']
            excel.range((row, column+13)).value = metrics[c]['F-Score']
            excel.range((row, column+21)).value = metrics[c]['Precision']
            excel.range((row, column+29)).value = metrics[c]['Recall']
        column += 1
    return