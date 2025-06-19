import os
from PIL import Image
import pydicom
import numpy as np
from tqdm import tqdm

INPUT_FOLDER = 'inputs/LIDC-IDRI/stage2_test/LIDC-IDRI-0121'
OUTPUT_FOLDER = 'outputs/original/'
TARGET_SIZE = (128, 128)

# Crear carpeta de salida y limpiarla si ya existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for f in os.listdir(OUTPUT_FOLDER):
    file_path = os.path.join(OUTPUT_FOLDER, f)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error al eliminar {file_path}: {e}")

def load_dcm_image(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)

    # Convertir a Hounsfield Units (HU)
    intercept = dicom.get('RescaleIntercept', 0.0)
    slope = dicom.get('RescaleSlope', 1.0)
    img = img * slope + intercept

    # Aplicar ventana: centro -600, ancho 1500 → [-1350, 150]
    min_val = -1350
    max_val = 150
    img = np.clip(img, min_val, max_val)
    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return Image.fromarray(img)

def main():
    dcm_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.dcm')]
    print(f"Found {len(dcm_files)} DICOM files")

    for file in tqdm(sorted(dcm_files), desc="Processing DICOM images"):
        try:
            img = load_dcm_image(os.path.join(INPUT_FOLDER, file))
            img = img.resize(TARGET_SIZE, Image.NEAREST)
            img.save(os.path.join(OUTPUT_FOLDER, os.path.splitext(file)[0] + '.png'))
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == '__main__':
    main()
