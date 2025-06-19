import os
import numpy as np
from glob import glob
from pathlib import Path
from lungmask import mask
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

# Set paths
input_base = 'inputs/LIDC-IDRI/stage1_train/images'
output_base = 'inputs/processed_data_512'
output_images_train = os.path.join(output_base, 'images/train')
output_images_val = os.path.join(output_base, 'images/val')
output_masks_train = os.path.join(output_base, 'masks/train/0')
output_masks_val = os.path.join(output_base, 'masks/val/0')

# Create output directories
for path in [output_images_train, output_images_val, output_masks_train, output_masks_val]:
    os.makedirs(path, exist_ok=True)

def load_dicom_series(dicom_dir):
    """Load DICOM series using SimpleITK"""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        dicom_names = [f for f in dicom_names if not f.lower().endswith('.xml')]
        if not dicom_names:
            return None, []
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image, dicom_names
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        return None, []

def normalize_dicom_image(image_array):
    """Normalize DICOM image array"""
    window_center = -600
    window_width = 1500
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    image_array = np.clip(image_array, min_value, max_value)
    image_array = ((image_array - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    image_array = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
    return image_array

# Get list of patient folders
all_folders = [f'LIDC-IDRI-{i:04d}' for i in range(1, 121)]

# Split into train/val patients
train_patients, val_patients = train_test_split(all_folders, test_size=0.2, random_state=42)
print(f"📚 Pacientes en entrenamiento: {len(train_patients)}")
print(f"📊 Pacientes en validación: {len(val_patients)}")

# Start processing
file_counter = 1

for patient_list, img_output_path, mask_output_path, tag in [
    (train_patients, output_images_train, output_masks_train, "Train"),
    (val_patients, output_images_val, output_masks_val, "Val")
]:
    for folder_name in patient_list:
        folder_path = os.path.join(input_base, folder_name)
        if not os.path.exists(folder_path):
            continue
        print(f"\n📂 Procesando {tag}: {folder_name}")
        try:
            dicom_image, dicom_files = load_dicom_series(folder_path)
            if dicom_image is None:
                continue
            series_array = sitk.GetArrayFromImage(dicom_image)
            lung_mask_3d = mask.apply(dicom_image)
            for slice_idx in tqdm(range(series_array.shape[0]), desc=folder_name, unit="slice"):
                image_slice = normalize_dicom_image(series_array[slice_idx])
                mask_slice = lung_mask_3d[slice_idx].astype(np.uint8)
                if np.sum(mask_slice) < 1000:
                    continue
                mask_slice = (mask_slice > 0).astype(np.uint8) * 255
                filename = f"{file_counter:05d}.png"
                Image.fromarray(image_slice).save(os.path.join(img_output_path, filename))
                Image.fromarray(mask_slice).save(os.path.join(mask_output_path, filename))
                file_counter += 1
        except Exception as e:
            print(f"Error en paciente {folder_name}: {e}")
            continue

print(f"\n✅ Procesamiento completo. Total de imágenes procesadas: {file_counter - 1}")
