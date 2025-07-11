import os
import sys
import torch
import yaml
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from archs import NestedUNet

# --- Configuración global ---
CONFIG_PATH = 'models/processed_data_512_NestedUNet_binary_wDS/config.yml'
WEIGHTS_PATH = 'models/processed_data_512_NestedUNet_binary_wDS/best_model.pth'
OUTPUT_FOLDER = 'outputs/nested/'

# --- Funciones utilitarias ---
def load_config(path):
  with open(path, 'r') as f:
    return yaml.safe_load(f)

def load_model(weights_path, config, device):
  model = NestedUNet(
    input_channels=config.get('input_channels', 1),
    deep_supervision=config.get('deep_supervision', True)
  )
  model.load_state_dict(torch.load(weights_path, map_location=device))
  model.to(device)
  model.eval()
  return model

def get_img_ids(subset, config):
  img_path = os.path.normpath(os.path.join('outputs', 'original', f"*{config['img_ext']}"))
  img_files = glob(img_path)
  img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_files]

  print(f"📁 Buscando imágenes en: {img_path}")
  print(f"🖼️ Imágenes encontradas en {subset}: {len(img_ids)}")
  if len(img_ids) == 0:
    print(f"❌ No se encontraron imágenes en '{subset}'. Verifica la ruta.")
  else:
    print(f"✅ {subset.capitalize()} cargado correctamente")
    print(f"   Ejemplo de archivos: {img_ids[:3]}")
  return img_ids

# --- Dataset para test ---
class TestDataset(torch.utils.data.Dataset):
  def __init__(self, img_ids, img_dir, img_ext, transform):
    self.img_ids = img_ids
    self.img_dir = img_dir
    self.img_ext = img_ext
    self.transform = transform

  def __len__(self):
    return len(self.img_ids)

  def __getitem__(self, idx):
    img_id = self.img_ids[idx]
    img_path = os.path.join(self.img_dir, img_id + self.img_ext)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
      raise ValueError(f"Error al leer imagen: {img_path}")
    img = img.astype(np.float32) / 255.0
    img = img[..., None]
    augmented = self.transform(image=img)
    img_tensor = augmented['image']
    return img_tensor, img_id

# --- Predicción ---
def predict(model, input_tensor, device):
  with torch.no_grad():
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    if isinstance(output, list):
      output = output[-1]
    output = torch.sigmoid(output)
    pred = (output > 0.5).float()
    return pred

# --- Main ---
def main():
  config = load_config(CONFIG_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = load_model(WEIGHTS_PATH, config, device)

  test_img_ids = get_img_ids("test", config)

  test_transform = A.Compose([
    A.Resize(config['input_h'], config['input_w']),
    A.Normalize(mean=(0.0,), std=(1.0,)),
    ToTensorV2()
  ])

  test_dataset = TestDataset(
    img_ids=test_img_ids,
    img_dir=os.path.join('outputs', 'original'),
    img_ext=config['img_ext'],
    transform=test_transform
  )

  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    drop_last=False,
    pin_memory=True
  )

  os.makedirs(OUTPUT_FOLDER, exist_ok=True)

  print(f"🚀 Generando segmentaciones para {len(test_dataset)} imágenes...")
  for inputs, ids in tqdm(test_loader, desc="Guardando máscaras"):
    preds = predict(model, inputs, device)
    for i in range(preds.shape[0]):
      mask_np = preds[i, 0].cpu().numpy().astype(np.uint8) * 255
      save_path = os.path.join(OUTPUT_FOLDER, ids[i] + ".png")
      cv2.imwrite(save_path, mask_np)

  print(f"✅ Segmentaciones guardadas en: {OUTPUT_FOLDER}")
if __name__ == "__main__":
  main()
