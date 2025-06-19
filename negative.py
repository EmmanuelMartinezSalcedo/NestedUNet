import os
from PIL import Image
import numpy as np

input_dir = "inputs/processed_data_512/masks/0"
output_dir = "inputs/processed_data_512/masks/1"

# Crea el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Procesa cada imagen en la carpeta de máscaras positivas
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Abre la imagen y conviértela a array
        mask = Image.open(input_path).convert("L")  # escala de grises
        mask_array = np.array(mask)

        # Invierte los valores: 255 → 0, 0 → 255
        negative_mask = 255 - mask_array

        # Guarda la máscara negativa
        Image.fromarray(negative_mask.astype(np.uint8)).save(output_path)

print("Máscaras negativas generadas en:", output_dir)
