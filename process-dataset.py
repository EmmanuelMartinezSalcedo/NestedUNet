import os
import numpy as np
from lungmask import mask
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import cv2
from glob import glob
import shutil

# Configuración de rutas
preprocessed_dir = "preprocessed/LIDC-IDRI"
output_dir = "processed/LIDC-IDRI"
train_dir = os.path.join(preprocessed_dir, "stage1-train")
test_dir = os.path.join(preprocessed_dir, "stage2-test")

# Crear directorios de salida
for path in [
    os.path.join(output_dir, "stage1-train", "image"),
    os.path.join(output_dir, "stage1-train", "mask"),
    os.path.join(output_dir, "stage2-test", "image"),
    os.path.join(output_dir, "stage2-test", "mask")
]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_dicom_series(dicom_dir):
    """Cargar una serie DICOM usando SimpleITK"""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        # Filtrar archivos que no sean DICOM (como .xml)
        dicom_names = [f for f in dicom_names if not f.lower().endswith('.xml')]
        if not dicom_names:
            return None, []
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image, dicom_names
    except Exception as e:
        print(f"Error cargando DICOM: {e}")
        return None, []

def normalize_dicom_image(image_array):
    """Normalizar imagen DICOM a 0-255 con ventana de pulmón"""
    window_center = -600
    window_width = 1500
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    image_array = np.clip(image_array, min_value, max_value)
    image_array = ((image_array - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    # Aplicar morfología para mejorar la imagen
    kernel = np.ones((3, 3), np.uint8)
    image_array = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
    return image_array

def process_dataset(source_dir, phase_name):
    """Procesar un conjunto de datos (train o test)"""
    print(f"\n{'='*60}")
    print(f"Procesando {phase_name}...")
    print(f"{'='*60}")
    
    # Obtener lista de pacientes
    patients = sorted([d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d))])
    
    file_counter = 1  # Contador que reinicia para cada stage
    
    for patient_idx, patient in enumerate(patients):
        patient_path = os.path.join(source_dir, patient)
        print(f"\n[{patient_idx+1}/{len(patients)}] Procesando paciente: {patient}")
        
        try:
            # Cargar serie DICOM
            dicom_image, dicom_files = load_dicom_series(patient_path)
            if dicom_image is None:
                print(f"  ⚠ No se encontraron imágenes DICOM válidas")
                continue
                
            print(f"  Cargados {len(dicom_files)} archivos DICOM")
            
            # Obtener array 3D
            series_array = sitk.GetArrayFromImage(dicom_image)
            
            # Generar máscara de pulmones con lungmask
            print(f"  Generando máscaras con lungmask...")
            lung_mask_3d = mask.apply(dicom_image)
            print(f"  ✓ Máscaras generadas")
            
            # Procesar cada slice
            for slice_idx in tqdm(range(series_array.shape[0]), desc=f"Slice {patient}", unit="slice"):
                # Normalizar imagen
                image_slice = normalize_dicom_image(series_array[slice_idx])
                
                # Procesar máscara
                mask_slice = lung_mask_3d[slice_idx].astype(np.uint8)
                # Convertir a binaria (0 o 255)
                mask_slice = (mask_slice > 0).astype(np.uint8) * 255
                
                # Generar nombre de archivo con número secuencial
                filename = f"{file_counter:06d}.png"
                
                # Guardar imagen procesada
                img_path = os.path.join(output_dir, phase_name, "image", filename)
                Image.fromarray(image_slice).save(img_path)
                
                # Guardar máscara
                mask_path = os.path.join(output_dir, phase_name, "mask", filename)
                Image.fromarray(mask_slice).save(mask_path)
                
                file_counter += 1
                
        except Exception as e:
            print(f"  ⚠ Error en paciente {patient}: {e}")
            continue
    
    return file_counter - 1  # Retornar el total de archivos procesados

# Procesar train
total_train = process_dataset(train_dir, "stage1-train")

# Procesar test
total_test = process_dataset(test_dir, "stage2-test")

print(f"\n{'='*60}")
print("Proceso completado exitosamente")
print(f"{'='*60}")
print(f"\nResumen:")
print(f"  Train: {total_train} imágenes")
print(f"    - Imágenes: {output_dir}/stage1-train/image/")
print(f"    - Máscaras: {output_dir}/stage1-train/mask/")
print(f"  Test: {total_test} imágenes")
print(f"    - Imágenes: {output_dir}/stage2-test/image/")
print(f"    - Máscaras: {output_dir}/stage2-test/mask/")
print(f"\nEstructura final:")
print(f"  processed/LIDC-IDRI/stage1-train/image/000001.png ... {total_train:06d}.png")
print(f"  processed/LIDC-IDRI/stage1-train/mask/000001.png ... {total_train:06d}.png")
print(f"  processed/LIDC-IDRI/stage2-test/image/000001.png ... {total_test:06d}.png")
print(f"  processed/LIDC-IDRI/stage2-test/mask/000001.png ... {total_test:06d}.png")