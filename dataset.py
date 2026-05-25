import os
import sys
from pathlib import Path
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# 1. Ensure current directory is in system path for clean imports
sys.path.append(str(Path(__file__).parent))

# Import your helper utilities from your exported .py file
try:
    from generate_and_decode_heatmaps import generate_heatmaps, decode_image, resize_image
except ImportError:
    raise ImportError(
        "Could not find 'generate_and_decode_heatmaps.py'. "
        "Please run 'jupyter nbconvert --to script generate_and_decode_heatmaps.ipynb' first!"
    )

# =====================================================================
# DATASET CLASS DEFINITION
# =====================================================================
class CatCenterNetDataset(Dataset):
    def __init__(self, image_paths, sigma=1.5, mean=None, std=None):
        self.image_paths = list(image_paths)
        self.sigma = sigma
        
        # If mean and std are provided, perform full normalization.
        # Otherwise, only apply ToTensor (critical for running the stats calculator).
        if mean is not None and std is not None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load and resize the image input (Output shape: 160x224x3)
        image_resized = resize_image(image_path)
        
        # Generate the list of 5 ground-truth numpy heatmaps (Each shape: 40x56)
        heatmaps_list = generate_heatmaps(image_path, sigma=self.sigma)
        
        # Convert the resized image to a normalized PyTorch tensor [3, 160, 224]
        image_tensor = self.transform(image_resized)
        
        # Stack the 5 separate 2D heatmaps into a single 3D tensor [5, 40, 56]
        target_tensor = torch.stack([torch.from_numpy(m) for m in heatmaps_list], dim=0).float()
        
        return image_tensor, target_tensor

    @staticmethod
    def calculate_stats(image_paths, batch_size=32, num_workers=2):
        """
        Statically computes the precise per-channel mean and standard deviation 
        of the dataset using an un-normalized temporary instance.
        """
        # Create a temporary un-normalized instance of this dataset class
        temp_dataset = CatCenterNetDataset(image_paths, mean=None, std=None)
        loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        print(f"Calculating stats across {len(image_paths)} images...")
        for images, _ in tqdm(loader):
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            
            # Sum pixels across batch, height, and width coordinates
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
            
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

        mean = fst_moment.tolist()
        std = torch.sqrt(snd_moment - fst_moment ** 2).tolist()
        
        print(f"\nCalculation Complete!")
        return mean, std

def main():
    # Define your crop folder path matching your VS Code tree
    dataset_dir = Path("dataset/crop_v2")
    
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Could not locate directory at: '{dataset_dir.resolve()}'. "
            f"Please run this script from your main CatPlayground folder workspace."
        )
        
    # Gather all available image files
    all_images = sorted(list(dataset_dir.glob("*.png")))
    
    if len(all_images) == 0:
        raise ValueError(f"No .png images found inside the target folder: '{dataset_dir}'")
        
    print(f"Found {len(all_images)} total images to analyze.")
    
    # Shuffle to get an unbiased sequence distribution
    random.seed(42)
    random.shuffle(all_images)
    
    try:
        # Run the internal static calculation method
        calculated_mean, calculated_std = CatCenterNetDataset.calculate_stats(
            image_paths=all_images,
            batch_size=32,
            num_workers=2
        )
        
        print("=" * 60)
        print("         PRODUCTION READY DATASET CONFIGURATIONS         ")
        print("=" * 60)
        print(f"Copy and paste these arrays straight into your training loader init:")
        print(f"  mean = {calculated_mean}")
        print(f"  std  = {calculated_std}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[Error] Calculation crashed: {e}")
        print("Tip: Make sure the label parser inside your notebook points to the correct JSON annotations path!")

if __name__ == "__main__":
    main()


#       mean = [0.6238875389099121, 0.5799762010574341, 0.5467942357063293]
#   std  = [0.11096493899822235, 0.1462928056716919, 0.18701171875]