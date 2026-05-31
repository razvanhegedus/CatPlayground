import cv2
import numpy as np



camera_points = np.array([
    [784, 211],   # top-right
    [486, 225],   # top-left
    [1267, 609],   # Bottom-right

    [451, 632],   # Middle-left
    [739, 211],   # Middle-right

    [744, 260],   # Top-left
    [1134, 513],   # Top-center
    [791, 236]    # Top-right
], dtype=np.float32)



servo_points = np.array([
    [130, 75],    
    [119, 75],    
    [150, 90],  

    [118, 96],    
    [125, 75],    

    [125, 80],     
    [140, 90],    
    [130, 85]     
], dtype=np.float32)



H, mask = cv2.findHomography(
    camera_points,
    servo_points,
    cv2.RANSAC,
    3.0
)

print("Homography Matrix:")
print(H)


def pixel_to_servo(x, y):

    point = np.array([[[x, y]]], dtype=np.float32)

    transformed = cv2.perspectiveTransform(point, H)

    pan = transformed[0][0][0]
    tilt = transformed[0][0][1]

    return pan, tilt

test_points = [
    (750, 225),
    (250, 350),
    (400, 180)
]

for x, y in test_points:

    pan, tilt = pixel_to_servo(x, y)

    print(f"\nPixel: ({x}, {y})")
    print(f"Pan : {pan:.2f}")
    print(f"Tilt: {tilt:.2f}")



def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

# pan = clamp(pan, 0, 180)
# tilt = clamp(tilt, 0, 180)