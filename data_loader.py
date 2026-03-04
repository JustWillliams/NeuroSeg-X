import os
import zipfile
import shutil
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from google.colab import drive

def setup_colab_data(drive_path='/content/drive/MyDrive/Colab_Notebooks_Data'):
    """
    Mounts Google Drive and extracts the three brain datasets.
    """
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    datasets = ['brain_glioma.zip', 'brain_menin.zip', 'brain_tumor.zip']
    extract_base = '/content/brain_data'
    
    os.makedirs(extract_base, exist_ok=True)
    
    for ds in datasets:
        zip_path = os.path.join(drive_path, ds)
        if not os.path.exists(zip_path):
            print(f"Warning: {zip_path} not found!")
            continue
            
        print(f"Extracting {ds}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_base)
            
    print("Data extraction complete.")
    return extract_base

class NeuroSegDataset(Dataset):
    """
    Multi-task Dataset for NeuroSeg-X:
    - Segmentation (Masks)
    - Detection (Binary: Tumor Present)
    - Grading (Binary: LGG vs HGG)
    """
    def __init__(self, image_paths, mask_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels  # Dict with 'detection' and 'grading'
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]), dtype=np.float32)
        
        # Detection label (assuming mask exists means tumor present)
        det_label = self.labels['detection'][idx]
        
        # Grading label
        grad_label = self.labels['grading'][idx]
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return {
            'image': image,
            'mask': mask,
            'detection': torch.tensor(det_label, dtype=torch.float32),
            'grading': torch.tensor(grad_label, dtype=torch.long)
        }

def get_transforms(img_size=(512, 512)):
    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform
