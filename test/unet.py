import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm

# ---------------------
# Configuración
# ---------------------
INPUT_FOLDER = "processed/LIDC-IDRI/stage3-test/images"
OUTPUT_FOLDER = "outputs/unetpp_pretrained"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configuración del modelo
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['lesion']
ACTIVATION = 'sigmoid'

# Parámetros de procesamiento
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_SIZE = (256, 256)

# ---------------------
# Cargar modelo preentrenado
# ---------------------
def load_pretrained_model():
    model = UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    
    model = model.to(DEVICE)
    model.eval()
    
    # Obtener función de preprocesamiento específica para el codificador
    preprocess_input = get_preprocessing_fn(ENCODER, pretrained=ENCODER_WEIGHTS)
    
    return model, preprocess_input

# ---------------------
# Dataset para predicción (CORREGIDO)
# ---------------------
class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess_fn, transform=None):
        self.image_paths = image_paths
        self.preprocess_fn = preprocess_fn
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # Leer imagen
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        
        # Convertir a 3 canales (RGB) ya que el modelo preentrenado espera 3 canales
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Aplicar preprocesamiento específico del modelo
        # IMPORTANTE: El preprocesamiento devuelve float64, lo convertimos a float32
        img_preprocessed = self.preprocess_fn(img_rgb).astype(np.float32)
        
        # Aplicar transformaciones adicionales
        if self.transform:
            augmented = self.transform(image=img_preprocessed)
            img_tensor = augmented['image']
        else:
            img_tensor = ToTensorV2()(image=img_preprocessed)['image']
        
        return img_tensor, img_id

# ---------------------
# Función de predicción (CORREGIDA)
# ---------------------
def predict(model, dataloader, threshold=0.5):
    predictions = []
    img_ids = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(dataloader, desc="Prediciendo"):
            # Asegurarse de que los inputs sean float32
            inputs = inputs.to(DEVICE).float()
            
            # Realizar predicción
            outputs = model(inputs)
            
            # Aplicar umbral para binarización
            preds = (outputs > threshold).float()
            
            # Guardar resultados
            predictions.extend(preds.cpu().numpy())
            img_ids.extend(ids)
    
    return predictions, img_ids

# ---------------------
# Función principal
# ---------------------
def main():
    print("🚀 Iniciando predicción con UNet++ preentrenado...")
    
    # Cargar modelo preentrenado
    model, preprocess_fn = load_pretrained_model()
    print(f"✅ Modelo UNet++ con codificador {ENCODER} cargado")
    
    # Obtener lista de imágenes
    image_paths = [
        os.path.join(INPUT_FOLDER, f) 
        for f in os.listdir(INPUT_FOLDER) 
        if f.endswith('.png')
    ]
    
    if not image_paths:
        print("❌ No se encontraron imágenes en la carpeta de entrada")
        return
    
    print(f"📁 Se encontraron {len(image_paths)} imágenes para procesar")
    
    # Definir transformaciones
    transform = A.Compose([
        A.Resize(*INPUT_SIZE),
        ToTensorV2(),
    ])
    
    # Crear dataset y dataloader
    dataset = PredictionDataset(
        image_paths=image_paths,
        preprocess_fn=preprocess_fn,
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Realizar predicciones
    predictions, img_ids = predict(model, dataloader)
    
    # Guardar predicciones
    print("💾 Guardando predicciones...")
    for pred, img_id in zip(predictions, img_ids):
        # La predicción tiene forma [1, H, W], la convertimos a [H, W]
        mask = pred[0]
        
        # Convertir a uint8 y escalar a 0-255
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Guardar máscara
        output_path = os.path.join(OUTPUT_FOLDER, f"{img_id}.png")
        cv2.imwrite(output_path, mask_uint8)
    
    print(f"✅ Predicciones guardadas en: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()