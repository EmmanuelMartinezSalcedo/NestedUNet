import os
import numpy as np
from lungmask import mask
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import cv2
import random
import shutil

# Configuración
base_dir = "dataset/LIDC-IDRI"
output_dir = "processed/LIDC-IDRI"
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42

# Crear directorios de salida
for path in [
    os.path.join(output_dir, "stage1-train", "images"),
    os.path.join(output_dir, "stage1-train", "masks"),
    os.path.join(output_dir, "stage2-val", "images"),
    os.path.join(output_dir, "stage2-val", "masks"),
    os.path.join(output_dir, "stage3-test", "images"),
    os.path.join(output_dir, "stage3-test", "masks")
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

def process_patients(patients, phase_name, start_counter=1):
    """Procesar un conjunto de pacientes"""
    print(f"\n{'='*60}")
    print(f"Procesando {phase_name}...")
    print(f"{'='*60}")
    
    file_counter = start_counter
    
    for patient_idx, patient in enumerate(patients):
        # Ruta de la carpeta más interna (donde están los DCM)
        patient_path = os.path.join(base_dir, patient, patient, patient)
        
        if not os.path.exists(patient_path):
            print(f"  ⚠ No se encontró: {patient_path}")
            continue
            
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
            for slice_idx in tqdm(range(series_array.shape[0]), desc=f"  Slices {patient}", unit="slice"):
                # Normalizar imagen
                image_slice = normalize_dicom_image(series_array[slice_idx])
                
                # Procesar máscara
                mask_slice = lung_mask_3d[slice_idx].astype(np.uint8)
                # Convertir a binaria (0 o 255)
                mask_slice = (mask_slice > 0).astype(np.uint8) * 255
                
                # Generar nombre de archivo con número secuencial
                filename = f"{file_counter:06d}.png"
                
                # Guardar imagen procesada
                img_path = os.path.join(output_dir, phase_name, "images", filename)
                Image.fromarray(image_slice).save(img_path)
                
                # Guardar máscara
                mask_path = os.path.join(output_dir, phase_name, "masks", filename)
                Image.fromarray(mask_slice).save(mask_path)
                
                file_counter += 1
                
        except Exception as e:
            print(f"  ⚠ Error en paciente {patient}: {e}")
            continue
    
    return file_counter - start_counter  # Retornar el total de archivos procesados

# Obtener lista de pacientes
print("Escaneando pacientes...")
patients = []
for patient in sorted(os.listdir(base_dir)):
    patient_path = os.path.join(base_dir, patient)
    if os.path.isdir(patient_path):
        # Verificar que tenga la estructura correcta
        final_path = os.path.join(patient_path, patient, patient)
        if os.path.exists(final_path):
            patients.append(patient)

print(f"Total de pacientes encontrados: {len(patients)}")

# Mezclar y dividir en 3 conjuntos
random.seed(random_seed)
random.shuffle(patients)

train_split = int(len(patients) * train_ratio)
val_split = int(len(patients) * (train_ratio + val_ratio))

train_patients = patients[:train_split]
val_patients = patients[train_split:val_split]
test_patients = patients[val_split:]

print(f"\nDivisión de pacientes:")
print(f"  Train: {len(train_patients)} pacientes ({len(train_patients)/len(patients)*100:.1f}%)")
print(f"  Val:   {len(val_patients)} pacientes ({len(val_patients)/len(patients)*100:.1f}%)")
print(f"  Test:  {len(test_patients)} pacientes ({len(test_patients)/len(patients)*100:.1f}%)")

# Procesar cada conjunto
total_train = process_patients(train_patients, "stage1-train", start_counter=1)
total_val = process_patients(val_patients, "stage2-val", start_counter=1)
total_test = process_patients(test_patients, "stage3-test", start_counter=1)

print(f"\n{'='*60}")
print("Proceso completado exitosamente")
print(f"{'='*60}")
print(f"\nResumen:")
print(f"  Train: {total_train} imágenes ({len(train_patients)} pacientes)")
print(f"    - Imágenes: {output_dir}/stage1-train/images/")
print(f"    - Máscaras: {output_dir}/stage1-train/masks/")
print(f"  Val: {total_val} imágenes ({len(val_patients)} pacientes)")
print(f"    - Imágenes: {output_dir}/stage2-val/images/")
print(f"    - Máscaras: {output_dir}/stage2-val/masks/")
print(f"  Test: {total_test} imágenes ({len(test_patients)} pacientes)")
print(f"    - Imágenes: {output_dir}/stage3-test/images/")
print(f"    - Máscaras: {output_dir}/stage3-test/masks/")
print(f"\nEstructura final:")
print(f"  processed/LIDC-IDRI/stage1-train/images/000001.png ... {total_train:06d}.png")
print(f"  processed/LIDC-IDRI/stage1-train/masks/000001.png ... {total_train:06d}.png")
print(f"  processed/LIDC-IDRI/stage2-val/images/000001.png ... {total_val:06d}.png")
print(f"  processed/LIDC-IDRI/stage2-val/masks/000001.png ... {total_val:06d}.png")
print(f"  processed/LIDC-IDRI/stage3-test/images/000001.png ... {total_test:06d}.png")
print(f"  processed/LIDC-IDRI/stage3-test/masks/000001.png ... {total_test:06d}.png")