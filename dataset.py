import os
import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Construct image path
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        
        # Check if image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image was loaded successfully
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}. "
                           f"File may be corrupted or in unsupported format.")
        
        img = img[..., None]  # [H, W, 1]

        # Construct mask path
        mask_path = os.path.join(self.mask_dir, "0", img_id + self.mask_ext)
        
        # Check if mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if mask was loaded successfully
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}. "
                           f"File may be corrupted or in unsupported format.")
        
        mask = mask[..., None]  # [H, W, 1]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # [1, H, W]
        
        mask = mask.astype('float32') / 255.0  # [0, 1]
        mask = (mask > 0.5).astype('float32')
        mask = mask.transpose(2, 0, 1)  # [1, H, W]

        return img, mask, {'img_id': img_id}
    
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
            raise ValueError(f"Failed to load image: {img_path}. "
                             f"File may be corrupted or in unsupported format.")
        
        img = img[..., None]  # [H, W, 1]

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            img = img.astype('float32') / 255.0
            img = img.transpose(2, 0, 1)  # [1, H, W]
            img = torch.from_numpy(img)

        return img, {'img_id': img_id}