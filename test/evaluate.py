import os
import sys
import yaml
import torch
import shutil
import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
from lungmask import mask
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score


CONFIG_PATH = 'models/processed_data_512_NestedUNet_binary_wDS/config.yml'
WEIGHTS_PATH = 'models/processed_data_512_NestedUNet_binary_wDS/best_model.pth'

BASE_FOLDER = 'inputs/LIDC-IDRI/stage2_test/'
ORIGINAL_FOLDER = 'outputs/original/'
GROUND_FOLDER = 'outputs/groundtruth/'
NESTED_FOLDER = 'outputs/nested/'

TARGET_SIZE = (128, 128)

def should_generate_new_images():
  existing = any(
    os.listdir(folder)
    for folder in [ORIGINAL_FOLDER, GROUND_FOLDER, NESTED_FOLDER]
    if os.path.exists(folder)
  )
  if existing:
    response = input("¿Deseas generar nuevas imágenes y sobrescribir las anteriores? (y/n): ").strip().lower()
    return response == 'y'
  return True

def clear_output_folders():
  for folder in [ORIGINAL_FOLDER, GROUND_FOLDER, NESTED_FOLDER]:
    if os.path.exists(folder):
      shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def load_config(path):
  with open(path, 'r') as f:
    return yaml.safe_load(f)

def load_model(weights_path, config, device):
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  from archs import NestedUNet
  model = NestedUNet(
    input_channels=config.get('input_channels', 1),
    deep_supervision=config.get('deep_supervision', True)
  )
  model.load_state_dict(torch.load(weights_path, map_location=device))
  model.to(device)
  model.eval()
  return model

def predict(model, input_tensor, device):
  with torch.no_grad():
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    if isinstance(output, list):
      output = output[-1]
    output = torch.sigmoid(output)
    return (output > 0.5).float()

def binarize(mask):
  return (mask > 127).astype(np.uint8)

def dice_score(pred, target, smooth=1e-5):
  pred = binarize(pred)
  target = binarize(target)
  intersection = np.logical_and(pred, target).sum()
  return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-5):
  pred = binarize(pred)
  target = binarize(target)
  intersection = np.logical_and(pred, target).sum()
  union = np.logical_or(pred, target).sum()
  return (intersection + smooth) / (union + smooth)

def accuracy_score(pred, target):
  pred = binarize(pred)
  target = binarize(target)
  return (pred == target).sum() / pred.size

def precision_score(pred, target, smooth=1e-5):
  pred = binarize(pred)
  target = binarize(target)
  tp = np.logical_and(pred == 1, target == 1).sum()
  fp = np.logical_and(pred == 1, target == 0).sum()
  return (tp + smooth) / (tp + fp + smooth)

def recall_score(pred, target, smooth=1e-5):
  pred = binarize(pred)
  target = binarize(target)
  tp = np.logical_and(pred == 1, target == 1).sum()
  fn = np.logical_and(pred == 0, target == 1).sum()
  return (tp + smooth) / (tp + fn + smooth)

def specificity_score(pred, target, smooth=1e-5):
  pred = binarize(pred)
  target = binarize(target)
  tn = np.logical_and(pred == 0, target == 0).sum()
  fp = np.logical_and(pred == 1, target == 0).sum()
  return (tn + smooth) / (tn + fp + smooth)

def f1_score(pred, target, smooth=1e-5):
  prec = precision_score(pred, target, smooth)
  rec = recall_score(pred, target, smooth)
  return (2 * prec * rec + smooth) / (prec + rec + smooth)

def auc_score(pred, target):
  pred = pred.astype(np.float32) / 255.0 
  target = binarize(target).astype(np.uint8)
  if np.unique(target).size == 1:
    return np.nan
  return roc_auc_score(target.flatten(), pred.flatten())

def preprocess_slice(img, config, test_transform):
  img = np.clip(img, -1350, 150)
  img = ((img + 1350) / 1500).astype(np.float32)
  img = cv2.resize(img, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
  img = img[..., None]
  augmented = test_transform(image=img)
  return augmented['image'].unsqueeze(0)

def main():
  config = load_config(CONFIG_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  test_transform = A.Compose([
    A.Resize(config['input_h'], config['input_w']),
    A.Normalize(mean=(0.0,), std=(1.0,)),
    ToTensorV2()
  ])
  model = load_model(WEIGHTS_PATH, config, device)

  regenerate = should_generate_new_images()
  if regenerate:
    clear_output_folders()

  global_index = 1
  dice_vals, iou_vals = [], []
  acc_vals, prec_vals, rec_vals, f1_vals, spec_vals, auc_vals = [], [], [], [], [], []

  for folder in sorted(os.listdir(BASE_FOLDER)):
    folder_path = os.path.join(BASE_FOLDER, folder)
    if not os.path.isdir(folder_path):
      continue

    try:
      reader = sitk.ImageSeriesReader()
      dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
      if not dicom_names:
        continue
      reader.SetFileNames(dicom_names)
      volume = reader.Execute()
      volume_np = sitk.GetArrayFromImage(volume)
      gt_mask = mask.apply(volume)
      num_slices = gt_mask.shape[0]

      for i in range(num_slices):
        slice_idx = num_slices - 1 - i
        base_name = f"{global_index:06d}.png"

        if regenerate:
          orig_slice = volume_np[slice_idx]
          orig_proc = np.clip(orig_slice, -1350, 150)
          orig_proc = ((orig_proc + 1350) / 1500 * 255).astype(np.uint8)
          orig_img = Image.fromarray(orig_proc).resize(TARGET_SIZE, Image.NEAREST)
          orig_img.save(os.path.join(ORIGINAL_FOLDER, base_name))

          gt_slice = (gt_mask[slice_idx] > 0).astype(np.uint8) * 255
          gt_img = Image.fromarray(gt_slice).resize(TARGET_SIZE, Image.NEAREST)
          gt_img.save(os.path.join(GROUND_FOLDER, base_name))

          input_tensor = preprocess_slice(orig_slice, config, test_transform)
          pred_tensor = predict(model, input_tensor, device)
          pred = pred_tensor[0, 0].cpu().numpy()
          pred_img = (pred * 255).astype(np.uint8)
          pred_resized = cv2.resize(pred_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
          cv2.imwrite(os.path.join(NESTED_FOLDER, base_name), pred_resized)

        # Cargar desde disco
        pred_resized = cv2.imread(os.path.join(NESTED_FOLDER, base_name), cv2.IMREAD_GRAYSCALE)
        gt_array = cv2.imread(os.path.join(GROUND_FOLDER, base_name), cv2.IMREAD_GRAYSCALE)

        dice_vals.append(dice_score(pred_resized, gt_array))
        iou_vals.append(iou_score(pred_resized, gt_array))
        acc_vals.append(accuracy_score(pred_resized, gt_array))
        prec_vals.append(precision_score(pred_resized, gt_array))
        rec_vals.append(recall_score(pred_resized, gt_array))
        f1_vals.append(f1_score(pred_resized, gt_array))
        spec_vals.append(specificity_score(pred_resized, gt_array))
        auc_vals.append(auc_score(pred_resized, gt_array))

        global_index += 1

    except Exception:
      continue

  if dice_vals:
    print(f"Dice: {np.mean(dice_vals):.4f}")
    print(f"IoU: {np.mean(iou_vals):.4f}")
    print(f"Accuracy: {np.mean(acc_vals):.4f}")
    print(f"Precision: {np.mean(prec_vals):.4f}")
    print(f"Recall: {np.mean(rec_vals):.4f}")
    print(f"F1 Score: {np.mean(f1_vals):.4f}")
    print(f"Specificity: {np.mean(spec_vals):.4f}")
    print(f"AUC: {np.nanmean(auc_vals):.4f}")
  else:
    print("No se calcularon métricas.")

if __name__ == "__main__":
  main()
