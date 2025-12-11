import os
from PIL import Image
from tqdm import tqdm

IMG_INPUT = "processed/LIDC-IDRI/stage3-test/images"
MASK_INPUT = "processed/LIDC-IDRI/stage3-test/masks"

IMG_OUTPUT = "outputs/original"
MASK_OUTPUT = "outputs/groundtruth"

TARGET_SIZE = (128, 128)

# ---------------------------------------------------
# Crear carpetas y limpiarlas
# ---------------------------------------------------
os.makedirs(IMG_OUTPUT, exist_ok=True)
os.makedirs(MASK_OUTPUT, exist_ok=True)

for folder in [IMG_OUTPUT, MASK_OUTPUT]:
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            os.remove(path)

# ---------------------------------------------------
# Función para procesar imágenes
# ---------------------------------------------------
def process_folder(input_folder, output_folder):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]

    for file in tqdm(files, desc=f"Procesando {input_folder}"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        try:
            img = Image.open(input_path).convert("L")  # grayscale
            img = img.resize(TARGET_SIZE, Image.NEAREST)
            img.save(output_path)
        except Exception as e:
            print(f"Error procesando {file}: {e}")

# ---------------------------------------------------
# Procesar imágenes y máscaras
# ---------------------------------------------------
process_folder(IMG_INPUT, IMG_OUTPUT)
process_folder(MASK_INPUT, MASK_OUTPUT)

print("✔ Proceso completado correctamente.")
