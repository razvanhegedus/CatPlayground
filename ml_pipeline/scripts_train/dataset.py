from pathlib import Path

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ml_pipeline.scripts_train.generate_and_decode_heatmaps import generate_heatmaps, resize_image
import torch
import torchvision.transforms as transforms
import cv2
import random
import numpy as np

class CatCenterNetDataset(Dataset):
    def __init__(self, image_paths, scale_factor=4, sigma=2, img_size=(224, 160)):
        self.image_paths = list(image_paths)
        self.scale_factor = scale_factor
        self.sigma = sigma
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = resize_image(img_path)
            
        target_maps = generate_heatmaps(img_path, self.scale_factor, self.sigma)
        
        target_maps_np = np.array(target_maps)
        
        image_tensor = self.transform(image)
        target_tensor = torch.from_numpy(target_maps_np).float()
        
        return image_tensor, target_tensor
    

def get_data_loaders(dataset_dir, batch_size=16, split_ratio=(0.8, 0.1, 0.1)):
    all_images = sorted(list(Path(dataset_dir).glob("*.png")))
    
    if not all_images:
        raise ValueError(f"No .png images found in {dataset_dir}")
    
    random.seed(42)
    random.shuffle(all_images)
    
    num_images = len(all_images)
    train_end = int(num_images * split_ratio[0])
    val_end = train_end + int(num_images * split_ratio[1])
    
    train_paths = all_images[:train_end]
    val_paths = all_images[train_end:val_end]
    test_paths = all_images[val_end:]
    
    #print(f"Total Images: {num_images} | Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    
    train_dataset = CatCenterNetDataset(train_paths)
    val_dataset   = CatCenterNetDataset(val_paths)
    test_dataset  = CatCenterNetDataset(test_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders("dataset/crop", batch_size=16)
