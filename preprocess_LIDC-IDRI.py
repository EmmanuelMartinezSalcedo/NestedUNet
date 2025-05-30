import os
from pathlib import Path
import numpy as np
from lungmask import mask
import SimpleITK as sitk
from PIL import Image
import random
from tqdm import tqdm
import shutil
import cv2

def load_dicom_series(dicom_dir):
    """Load DICOM series using SimpleITK"""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        dicom_names = [f for f in dicom_names if not f.lower().endswith('.xml')]
        
        if not dicom_names:
            print(f"No DICOM files found in {dicom_dir}")
            return None, []
            
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image, dicom_names
        
    except Exception as e:
        print(f"Error loading DICOM with SimpleITK: {e}")
        return None, []

def normalize_dicom_image(image_array):
    """Normalize DICOM image array"""
    # Apply window/level normalization for CT images
    window_center = -600  # Typical for lung window
    window_width = 1500   # Typical for lung window
    
    min_value = window_center - window_width/2
    max_value = window_center + window_width/2
    
    # Clip values to the window range and normalize
    image_array = np.clip(image_array, min_value, max_value)
    image_array = ((image_array - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    image_array = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
    
    return image_array

# Set paths
input_base = 'inputs/LIDC-IDRI/stage1_train/images'
output_images = 'inputs/processed_data_512/images'
output_masks = 'inputs/processed_data_512/masks/0'

# Create output directories
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# Initialize counter for file naming
file_counter = 1
valid_pairs = []

# Process each folder
for folder_id in range(1, 121):
    folder_name = f'LIDC-IDRI-{folder_id:04d}'
    folder_path = os.path.join(input_base, folder_name)
    
    if not os.path.exists(folder_path):
        continue
    
    print(f"\nProcessing {folder_name}")
    
    try:
        # Load DICOM series
        dicom_image, dicom_files = load_dicom_series(folder_path)
        
        if dicom_image is None or not dicom_files:
            print(f"Could not load DICOM series from {folder_path}")
            continue
        
        # Get array from DICOM series
        series_array = sitk.GetArrayFromImage(dicom_image)
        num_slices = series_array.shape[0]
        
        # Apply lungmask to entire volume
        print(f"Applying lungmask to {num_slices} files...")
        lung_mask_3d = mask.apply(dicom_image)
        
        if lung_mask_3d is None:
            print(f"Could not generate mask for {folder_path}")
            continue
        
        # Process each slice with progress bar
        for slice_idx in tqdm(range(num_slices), desc=f"Processing {folder_name}", unit="files"):
            try:
                # Get and normalize the image slice
                image_slice = normalize_dicom_image(series_array[slice_idx])
                
                # Get and process mask slice
                mask_slice = lung_mask_3d[slice_idx].astype(np.uint8)
                
                # Skip if mask is empty or too small
                if np.sum(mask_slice) < 1000:
                    continue
                
                # Normalize mask to binary
                mask_slice = (mask_slice > 0).astype(np.uint8) * 255
                
                # Save image and mask
                filename = f'{file_counter:05d}.png'
                
                Image.fromarray(image_slice).save(os.path.join(output_images, filename))
                Image.fromarray(mask_slice).save(os.path.join(output_masks, filename))
                
                valid_pairs.append(file_counter)
                file_counter += 1
                
            except Exception as e:
                print(f"Error processing slice {slice_idx} from {folder_path}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error processing folder {folder_path}: {str(e)}")
        continue

print(f"Generated {len(valid_pairs)} valid pairs")

# Shuffle files while maintaining pairs
if valid_pairs:
    print("Shuffling files...")
    new_indices = list(range(1, len(valid_pairs) + 1))
    random.shuffle(new_indices)
    
    # Create temporary directories
    temp_images = output_images + '_temp'
    temp_masks = output_masks + '_temp'
    os.makedirs(temp_images, exist_ok=True)
    os.makedirs(temp_masks, exist_ok=True)
    
    # Move files with new names
    for old_idx, new_idx in tqdm(zip(valid_pairs, new_indices), desc="Shuffling files"):
        old_name = f'{old_idx:05d}.png'
        new_name = f'{new_idx:05d}.png'
        
        shutil.move(os.path.join(output_images, old_name),
                   os.path.join(temp_images, new_name))
        shutil.move(os.path.join(output_masks, old_name),
                   os.path.join(temp_masks, new_name))
    
    # Replace original directories with shuffled ones
    shutil.rmtree(output_images)
    shutil.rmtree(output_masks)
    os.rename(temp_images, output_images)
    os.rename(temp_masks, output_masks)
    
print("Processing complete!")