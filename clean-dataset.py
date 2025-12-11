# PASO 1: Limpiar dataset conservando solo dcm

import os
import shutil

base_dir = "dataset/LIDC-IDRI"

for patient in sorted(os.listdir(base_dir)):
    patient_path = os.path.join(base_dir, patient)
    if not os.path.isdir(patient_path):
        continue

    print(f"\nProcesando {patient}...")

    correct_folder = None
    correct_level1 = None
    
    for level1_name in os.listdir(patient_path):
        level1_path = os.path.join(patient_path, level1_name)
        if not os.path.isdir(level1_path):
            continue
        
        for level2_name in os.listdir(level1_path):
            level2_path = os.path.join(level1_path, level2_name)
            if not os.path.isdir(level2_path):
                continue
            
            dicom_path = os.path.join(level2_path, "1-001.dcm")
            if os.path.exists(dicom_path):
                correct_folder = level2_path
                correct_level1 = level1_path
                print(f"  Carpeta correcta detectada: {level2_path}")
                break
        
        if correct_folder:
            break

    if not correct_folder:
        print("  No se encontró carpeta con 1-001.dcm, se omite.")
        continue

    xml_count = 0
    for file in os.listdir(correct_folder):
        if file.endswith(".xml"):
            os.remove(os.path.join(correct_folder, file))
            xml_count += 1
    if xml_count > 0:
        print(f"  {xml_count} archivo(s) XML eliminado(s)")

    for level1_name in os.listdir(patient_path):
        level1_path = os.path.join(patient_path, level1_name)
        if os.path.isdir(level1_path) and level1_path != correct_level1:
            shutil.rmtree(level1_path)
            print(f"  Carpeta incorrecta eliminada: {level1_name}")

    new_level2_name = patient
    new_level2_path = os.path.join(correct_level1, new_level2_name)
    if correct_folder != new_level2_path:
        os.rename(correct_folder, new_level2_path)
        print(f"  Renombrado nivel 2: {os.path.basename(correct_folder)} → {patient}")
        correct_folder = new_level2_path

    # Renombrar nivel 1
    new_level1_name = patient
    new_level1_path = os.path.join(patient_path, new_level1_name)
    if correct_level1 != new_level1_path:
        shutil.move(correct_level1, new_level1_path)
        print(f"  Renombrado nivel 1: {os.path.basename(correct_level1)} → {patient}")

    print(f"  Estructura finalizada para {patient}")
    final_path = os.path.join(patient_path, patient, patient)
    print(f"     Ruta final: {final_path}")

print("\n" + "="*50)
print("Proceso completado exitosamente.")
print("="*50)
