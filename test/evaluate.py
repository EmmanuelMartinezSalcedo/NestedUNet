import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

GROUNDTRUTH_FOLDER = "outputs/groundtruth"
PREDICTED_FOLDER = "outputs/predicted-LIDC-IDRI_NestedUNet_binary_woDS"  # <-- cámbialo a tu modelo


def load_mask(path):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return (2 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)


def iou_score(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-8)


def safe_auc(gt, pred):
    gt_flat = gt.flatten()
    if len(np.unique(gt_flat)) < 2:
        return None
    return roc_auc_score(gt_flat, pred.flatten())


def main():
    gt_files = sorted([f for f in os.listdir(GROUNDTRUTH_FOLDER) if f.endswith(".png")])
    pred_files = sorted([f for f in os.listdir(PREDICTED_FOLDER) if f.endswith(".png")])

    total = len(gt_files)

    dices, ious, aucs = [], [], []

    for idx, (gt_name, pred_name) in enumerate(zip(gt_files, pred_files), start=1):
        gt = load_mask(os.path.join(GROUNDTRUTH_FOLDER, gt_name))
        pred = load_mask(os.path.join(PREDICTED_FOLDER, pred_name))

        dices.append(dice_score(gt, pred))
        ious.append(iou_score(gt, pred))

        auc_val = safe_auc(gt, pred)
        if auc_val is not None:
            aucs.append(auc_val)

        # ---- Barra de progreso ----
        pct = (idx / total) * 100
        print(f"\rProcesando {idx}/{total} ({pct:.1f}%)", end="", flush=True)

    print("\n\n===== MÉTRICAS PROMEDIO =====")
    print(f"Dice Score: {np.mean(dices):.4f}")
    print(f"IoU Score : {np.mean(ious):.4f}")
    if len(aucs) > 0:
        print(f"AUC       : {np.mean(aucs):.4f}")
    else:
        print("AUC       : No calculable (ninguna imagen válida)")


if __name__ == "__main__":
    main()
