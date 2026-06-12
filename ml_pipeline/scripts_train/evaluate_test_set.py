import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

# --- 1. AUTOMATIC PATH RESOLUTION ---
# This automatically finds the CatPlayground root folder and adds it to the system path.
# This fixes the ModuleNotFoundError for both this script and dataset.py.
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent 

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- 2. LOCAL IMPORTS ---
from ml_pipeline.scripts_train.model import CatConvNet
from ml_pipeline.scripts_train.generate_and_decode_heatmaps import get_bbox_xyxy
from ml_pipeline.scripts_train.dataset import get_data_loaders

def decode_predictions(pred_dict, threshold=0.3, downscale_factor=4):
    """
    Decodes the dictionary output using the custom argmax and unravel logic.
    Processes a batch of predictions.
    """
    # 1. APPLY SIGMOID: Convert raw logits to probabilities (0.0 to 1.0)
    heatmap_probs = torch.sigmoid(pred_dict['heatmap'])
    
    batch_size = heatmap_probs.shape[0]
    pred_boxes = []
    pred_centers = []
    
    for i in range(batch_size):
        # 2. Use the activated probabilities for the center
        heatmap_center = heatmap_probs[i, 0]
        
        # Keep the sizes and offsets raw (they are regression targets, not probabilities)
        heatmap_size_x = pred_dict['size'][i, 0]
        heatmap_size_y = pred_dict['size'][i, 1]
        heatmap_offset_x = pred_dict['offset'][i, 0]
        heatmap_offset_y = pred_dict['offset'][i, 1]

        # 3. Find the peak center using your unravel_index logic
        y_tensor, x_tensor = torch.unravel_index(heatmap_center.argmax(), heatmap_center.shape)
        y_center_heatmap, x_center_heatmap = y_tensor.item(), x_tensor.item()
        
        # Now confidence is a true probability between 0 and 1!
        confidence = heatmap_center[y_center_heatmap, x_center_heatmap].item()
        
        if confidence >= threshold:
            # 4. Extract the size and offset values at that exact peak
            size_x   = heatmap_size_x[y_center_heatmap, x_center_heatmap].item()
            size_y   = heatmap_size_y[y_center_heatmap, x_center_heatmap].item()
            offset_x = heatmap_offset_x[y_center_heatmap, x_center_heatmap].item()
            offset_y = heatmap_offset_y[y_center_heatmap, x_center_heatmap].item()

            # --- DIAGNOSTIC PRINT ---
            # Uncomment this to see what sizes the model is actually predicting
            print(f"Raw Predicted Size -> W: {size_x:.2f}, H: {size_y:.2f}")

            # 5. Map the center back to the original image resolution
            center_x = (x_center_heatmap + offset_x) * downscale_factor
            center_y = (y_center_heatmap + offset_y) * downscale_factor

            # 6. Map the sizes back to the original image resolution (Enforce Positive)
            actual_size_x = abs(size_x) 
            actual_size_y = abs(size_y) 

            # 7. Reconstruct the bounding box and enforce valid geometry
            x_min = center_x - actual_size_x / 2
            y_min = center_y - actual_size_y / 2
            x_max = center_x + actual_size_x / 2
            y_max = center_y + actual_size_y / 2

            pred_boxes.append([min(x_min, x_max), min(y_min, y_max), max(x_min, x_max), max(y_min, y_max)])
            pred_centers.append([center_x, center_y])
        else:
            # If nothing passes the threshold, output a blank box
            pred_boxes.append([0.0, 0.0, 0.0, 0.0])
            pred_centers.append([0.0, 0.0])

    return torch.tensor(pred_boxes, dtype=torch.float32), torch.tensor(pred_centers, dtype=torch.float32)

def evaluate_test_set():
    # Accelerate inference on Apple Silicon
    device = torch.device('mps')
    print(f"Running evaluation on: {device}")

    # Load Model
    cat_model = CatConvNet()
    model_path = str(project_root / "ml_pipeline" / "weights" / "best_catnet_model.pth")
    cat_model.load_state_dict(torch.load(model_path, map_location=device))
    cat_model.to(device)
    cat_model.eval()

    # Load Data
    dataset_dir = str(project_root / "ml_pipeline" / "dataset" / "crop")
    
    try:
        _, _, test_loader = get_data_loaders(dataset_dir, batch_size=16)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    total_iou = 0.0
    valid_predictions = 0
    all_ious = []
    
    # NEW: List to track the Euclidean distances between centers
    center_errors_original_space = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            
            # Forward pass
            predictions = cat_model(images)
            
            # Move each individual tensor in the dictionary to the CPU
            for key in predictions.keys():
                predictions[key] = predictions[key].cpu()
            
            # Decode the predicted bounding boxes and centers
            pred_boxes, pred_centers = decode_predictions(predictions, threshold=0.3, downscale_factor=4)
            
            # Fetch the exact Ground Truth boxes for this batch
            start_idx = batch_idx * test_loader.batch_size
            batch_paths = test_loader.dataset.image_paths[start_idx : start_idx + len(images)]
            
            gt_boxes = []
            for path in batch_paths:
                down_left, up_right = get_bbox_xyxy(path) 
                
                # DIVIDE BY 4: Map the raw JSON coordinates down to the 160x224 network space
                # Enforce valid geometry on ground truth boxes
                x1 = down_left[0] / 4.0
                y1 = down_left[1] / 4.0
                x2 = up_right[0] / 4.0
                y2 = up_right[1] / 4.0
                
                gt_boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

            # Calculate IoU for the batch
            ious = torch.diag(box_iou(pred_boxes, gt_boxes))
            
            for j, iou_score in enumerate(ious):
                all_ious.append(iou_score.item())
                if iou_score > 0:
                    total_iou += iou_score.item()
                    valid_predictions += 1
                
                # --- NEW: Calculate Spatial Keypoint Distance Error ---
                px, py = pred_centers[j]
                
                # Only calculate distance if a valid prediction was made
                if px > 0.0 or py > 0.0:
                    gx_min, gy_min, gx_max, gy_max = gt_boxes[j]
                    
                    # Derive ground truth center from the bounding box
                    gx = (gx_min + gx_max) / 2.0
                    gy = (gy_min + gy_max) / 2.0
                    
                    # Calculate Euclidean distance
                    dist = torch.sqrt((px - gx)**2 + (py - gy)**2).item()
                    center_errors_original_space.append(dist)

    # Print summary metrics
    if not all_ious:
        print("No images were evaluated. Check your test set split ratio.")
        return

    all_ious_tensor = torch.tensor(all_ious)
    mean_iou = total_iou / valid_predictions if valid_predictions > 0 else 0.0
    ap_50_count = (all_ious_tensor >= 0.5).sum().item()
    
    # Calculate Mean Absolute Error (MAE) for centers
    if center_errors_original_space:
        mae_original = sum(center_errors_original_space) / len(center_errors_original_space)
        mae_feature_map = mae_original / 4.0  # Divide by downscale factor
    else:
        mae_original = 0.0
        mae_feature_map = 0.0
    
    print("\n--- Testing Set Evaluation Complete ---")
    print(f"Total Images Evaluated: {len(all_ious)}")
    print(f"Bounding Box mIoU: {mean_iou:.4f}")
    print(f"Bounding Box AP@50: {(ap_50_count / len(all_ious)) * 100:.1f}%")
    print(f"--- Spatial Keypoint Error ---")
    print(f"MAE (40x56 Feature Map): {mae_feature_map:.2f} pixels")
    print(f"MAE (160x224 Input Space): {mae_original:.2f} pixels")


if __name__ == "__main__":
    evaluate_test_set()