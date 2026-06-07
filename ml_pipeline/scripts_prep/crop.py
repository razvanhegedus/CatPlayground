from pathlib import Path
from PIL import Image
import numpy as np
import fiftyone as fo

def main():
    dataset_path = Path("/Users/hegedusrazvan/Developer/Faculty/IOTCA/CatPlayground/ml_pipeline/dataset/images")
    output_path = dataset_path.parent / "crop" 
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_name = "cat-playground-dataset"
    if not fo.dataset_exists(dataset_name):
        print(f"Error: FiftyOne dataset '{dataset_name}' does not exist.")
        return
        
    print(f"Loading '{dataset_name}' from FiftyOne...")
    fiftyone_dataset = fo.load_dataset(dataset_name)
    
    valid_fiftyone_names = {
        Path(sample.filepath).name.replace("frame_", "") 
        for sample in fiftyone_dataset
    }
    print(f"Found {len(valid_fiftyone_names)} valid frames registered in FiftyOne.")

    images = dataset_path.glob("*.png")
    images = sorted([img for img in images], key=lambda x: x.name)
    
    cropped_count = 0
    
    for image_path in images:
        normalized_local_name = image_path.name.replace("frame_", "")
    
        if normalized_local_name not in valid_fiftyone_names:
            continue
            
        with Image.open(image_path) as img:
            img_np = np.asarray(img)
            
            img_crop = img_np[80:, 384:, :]
            
            img_crop_pil = Image.fromarray(img_crop)
            
            # Saves keeping the original 'frame_xxxxx.png' format intact
            save_to = output_path / image_path.name
            img_crop_pil.save(save_to)
            cropped_count += 1
            
    print(f"Done! Successfully cropped {cropped_count} matching images.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()