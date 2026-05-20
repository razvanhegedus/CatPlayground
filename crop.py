from pathlib import Path
from PIL import Image
import numpy as np

def main():
    # Define paths
    dataset_path = Path("/Users/hegedusrazvan/Developer/Faculty/IOTCA/CatPlayground/dataset/images")
    output_path = dataset_path.parent / "crop" 
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = dataset_path.glob("*.png")
    images = sorted([img for img in images], key=lambda x: x.name)
    
    for image_path in images:
        with Image.open(image_path) as img:
            img_np = np.asarray(img)
            
            img_crop = img_np[80:, 400:, :]
            
            img_crop_pil = Image.fromarray(img_crop)
            
            save_to = output_path / image_path.name
            img_crop_pil.save(save_to)
            
    print(f"Done! Cropped images saved to: {output_path}")

if __name__ == "__main__":
    main()