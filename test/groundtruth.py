import os
from PIL import Image
import SimpleITK as sitk
import numpy as np
from lungmask import mask
from tqdm import tqdm
 
INPUT_FOLDER = 'inputs/LIDC-IDRI/stage2_test/LIDC-IDRI-0121'
OUTPUT_FOLDER = 'outputs/ground/'
ORIGINAL_FOLDER = 'outputs/original/'
TARGET_SIZE = (128, 128)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for f in os.listdir(OUTPUT_FOLDER):
    file_path = os.path.join(OUTPUT_FOLDER, f)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error al eliminar {file_path}: {e}")

def load_dicom_volume(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    return reader.Execute()

def main():
    print("Loading DICOM volume...")
    image = load_dicom_volume(INPUT_FOLDER)
    mask_3d = mask.apply(image)
    num_slices = mask_3d.shape[0]

    print(f"Saving masks for {num_slices} slices...")
    for i in tqdm(range(num_slices)):
        slice_idx = num_slices - 1 - i
        mask_slice = (mask_3d[slice_idx] > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_slice).resize(TARGET_SIZE, Image.NEAREST)
        mask_img.save(os.path.join(OUTPUT_FOLDER, f"1-{i+1:03d}.png"))

if __name__ == '__main__':
    main()
