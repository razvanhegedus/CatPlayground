from picamera2 import Picamera2
import cv2
import os
import time
import glob

output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

existing_files = glob.glob(f"{output_dir}/*.png")

image_id = len(existing_files)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

num_frames = 6
duration = 2.0
interval = duration / num_frames

print("Capturing...")

for i in range(num_frames):
    frame = picam2.capture_array()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    filename = os.path.join(
        output_dir,
        f"frame_{image_id:05d}.png"
    )

    cv2.imwrite(filename, frame)

    print("Saved:", filename)

    image_id += 1

    time.sleep(interval)

picam2.close()

print("Done.")