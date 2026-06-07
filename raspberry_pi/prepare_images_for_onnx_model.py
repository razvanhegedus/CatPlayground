import cv2
import numpy as np
import copy

def preprocess_for_onnx(image_path, downscale_factor=4):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop = image[80:, 384:]
    
    height = crop.shape[0] // downscale_factor
    width = crop.shape[1] // downscale_factor
    image_resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

    # 1. Convert to [0, 1] range (Equates to transforms.ToTensor())
    image_float = image_resized.astype(np.float32) / 255.0
    
    # 2. Apply ImageNet Normalization (Equates to transforms.Normalize())
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (image_float - mean) / std
    
    # 3. HWC to CHW
    chw_layout = np.transpose(normalized, (2, 0, 1))
    
    # 4. Add batch dimension
    onnx_ready_tensor = np.expand_dims(chw_layout, axis=0)
    
    return onnx_ready_tensor

def decode_image_to_full_scale(heatmaps, crop_offsets=(384, 80)):

    heatmap_center   = 1 / (1 + np.exp(-heatmaps[0]))
    heatmap_size_x   = heatmaps[1]
    heatmap_size_y   = heatmaps[2]
    heatmap_offset_x = heatmaps[3]
    heatmap_offset_y = heatmaps[4]

    max_idx = np.argmax(heatmap_center)
    y_hm, x_hm = np.unravel_index(max_idx, heatmap_center.shape)
    confidence = heatmap_center[y_hm, x_hm]

    size_x_pred   = heatmap_size_x[y_hm, x_hm]
    size_y_pred   = heatmap_size_y[y_hm, x_hm]
    offset_x      = heatmap_offset_x[y_hm, x_hm]
    offset_y      = heatmap_offset_y[y_hm, x_hm]

    center_x_hm = x_hm + offset_x
    center_y_hm = y_hm + offset_y

  

    center_x_crop = center_x_hm * 16
    center_y_crop = center_y_hm * 16
    
    size_x_crop = size_x_pred * 4
    size_y_crop = size_y_pred * 4

    shift_x, shift_y = crop_offsets
    
    center_x_full = center_x_crop + shift_x
    center_y_full = center_y_crop + shift_y
    print(f"{center_x_full}, {center_y_full}")
    print(f"{size_x_crop}, {size_y_crop}")
    print(f"{offset_x}, {offset_y}")

    bl = (int(center_x_full - size_x_crop / 2), int(center_y_full - size_y_crop / 2))
    tr = (int(center_x_full + size_x_crop / 2), int(center_y_full + size_y_crop / 2))

    return [bl, tr], confidence

def draw_rectangle_on_image(image_src, down_left, up_right):
    image = copy.deepcopy(image_src)
    up_left = (down_left[0], up_right[1])
    down_right = (up_right[0], down_left[1])
    cv2.line(image, down_left, down_right, (0, 255, 0))
    cv2.line(image, up_right, down_right, (0, 255, 0))
    cv2.line(image, up_left, up_right, (0, 255, 0))
    cv2.line(image, up_left, down_left, (0, 255, 0))
    

    return image

def reconstruct_image_from_heatmaps(image_src, points):
    bl, tr = points[0], points[1]
    image = np.ascontiguousarray(copy.deepcopy(image_src)) 

    image = draw_rectangle_on_image(image, bl, tr)
    return image
