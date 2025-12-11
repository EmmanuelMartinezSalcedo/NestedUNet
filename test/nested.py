import os
import sys
import yaml
import torch
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from archs import NestedUNet

# ---------------------
# TestDataset corregido
# ---------------------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Error al leer imagen: {img_path}")
        
        # Normalizar ANTES de la transformación
        img = img.astype(np.float32) / 255.0
        img = img[..., None]  # [H, W, 1]

        # Aplicar transformación SIN normalización adicional
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, img_id

# ---------------------
# RUTAS
# ---------------------
MODEL_DIR = "models/LIDC-IDRI_NestedUNet_binary_woDS"
CONFIG_PATH = f"{MODEL_DIR}/config.yml"
WEIGHTS_PATH = f"{MODEL_DIR}/checkpoints/model_epoch_20.pth"

INPUT_FOLDER = "processed/LIDC-IDRI/stage3-test/images"
OUTPUT_FOLDER = f"outputs/predicted-{os.path.basename(MODEL_DIR)}"

# ---------------------
# CONFIG & MODEL
# ---------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(weights_path, config, device):
    model = NestedUNet(
        input_channels=config.get("input_channels", 1),
        deep_supervision=config.get("deep_supervision", False)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# ---------------------
# MAIN
# ---------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("📄 Cargando configuración...")
    config = load_config(CONFIG_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Dispositivo: {device}")

    print("📦 Cargando modelo...")
    model = load_model(WEIGHTS_PATH, config, device)

    # Cargar imágenes
    img_ids = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(INPUT_FOLDER)
        if f.endswith(".png")
    ])

    print(f"🖼️ Imágenes encontradas: {len(img_ids)}")

    if len(img_ids) == 0:
        print("❌ No se encontraron imágenes.")
        return

    # Transformación CORREGIDA (sin A.Normalize)
    test_transform = A.Compose([
        A.Resize(config["input_h"], config["input_w"]),
        # ELIMINADO: A.Normalize(mean=(0.0,), std=(1.0,))
        ToTensorV2()
    ])

    test_dataset = TestDataset(
        img_ids=img_ids,
        img_dir=INPUT_FOLDER,
        img_ext=config["img_ext"],
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=True
    )

    print("🚀 Generando predicciones...")

    for inputs, img_ids_batch in tqdm(test_loader, desc="Procesando"):
        inputs = inputs.to(device)
        
        with torch.no_grad():
            out = model(inputs)
            if isinstance(out, list):
                out = out[-1]

            pred = torch.sigmoid(out)
            pred = (pred > 0.5).float()

        # Guardar predicciones
        for i in range(pred.shape[0]):
            mask_np = (pred[i, 0].cpu().numpy() * 255).astype(np.uint8)
            output_path = os.path.join(OUTPUT_FOLDER, f"{img_ids_batch[i]}.png")
            cv2.imwrite(output_path, mask_np)

    print(f"\n🎉 Predicciones guardadas en: {OUTPUT_FOLDER}\n")

if __name__ == "__main__":
    main()