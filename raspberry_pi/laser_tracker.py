

import time
import sys
import logging
import os
from pathlib import Path
import busio
import board
import cv2
import numpy as np
import onnxruntime as ort
from adafruit_servokit import ServoKit
from map_servo_coordinates_to_image import H, pixel_to_servo
from prepare_images_for_onnx_model import preprocess_for_onnx, decode_image_to_full_scale


# In[7]:


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# PCA9685/Servo Channels
PAN_CHANNEL = 0
TILT_CHANNEL = 1
LASER_CHANNEL = 4

# Smoothing factor (EMA alpha: 1.0 = no smoothing, lower = smoother but slower)
EMA_ALPHA = 0.85



# In[ ]:


class EdgeTracker:
    def __init__(self, model_path="cat_centernet.onnx"):
        self.model_path = model_path
        self.kit = None
        self.ort_session = None
        self.cap = None

        # State tracking
        self.last_pan = 90.0
        self.last_tilt = 90.0
        self.total_motion_duration = 0.0
        self.motion_start_time = None
        self.last_cx = FRAME_WIDTH / 2.0
        self.last_safe_y = FRAME_HEIGHT / 2.0

        # Initialize hardware and model
        self.init_hardware()
        self.init_inference_engine()
        #self.init_camera()

    def init_hardware(self):
        """Initializes the PCA9685 board with error handling."""
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            kit = ServoKit(channels=16, i2c=i2c)
            self.kit = kit
            self.kit.servo[PAN_CHANNEL].set_pulse_width_range(500, 2500)
            self.kit.servo[TILT_CHANNEL].set_pulse_width_range(500, 2500)
            self.kit.servo[PAN_CHANNEL].angle = self.last_pan
            self.kit.servo[TILT_CHANNEL].angle = self.last_tilt
            # Turn on laser (Channel 4 acting as digital output via duty cycle)
            self.kit._pca.channels[LASER_CHANNEL].duty_cycle = 0x0FFF
            logging.info("Hardware stack (PCA9685) initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to communicate with PCA9685 over I2C: {e}")
            sys.exit(1)

    def init_inference_engine(self):
        """Initializes ONNX Runtime using CPU options optimized for ARMv8."""
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2  # Balance performance on Pi 4B
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.ort_session = ort.InferenceSession(self.model_path, opts)
            self.input_name = self.ort_session.get_inputs()[0].name
            logging.info(f"ONNX Model '{self.model_path}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ONNX model: {e}")
            sys.exit(1)

    def init_camera(self):
        """Initializes the OpenCV Camera Stream."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            logging.error("Could not open video device.")
            sys.exit(1)
        logging.info("Camera stream initialized.")

    def preprocess_frame(self, frame):
        """Handles physical orientation flip and prepares array for ONNX."""
        return preprocess_for_onnx(frame)

    def calculate_safe_floor_angles(self, bl, tr, safety_multiplier=0.4):
        bl_x, bl_y = bl
        tr_x, tr_y = tr

        raw_cx = (bl_x + tr_x) / 2.0
        box_h = abs(bl_y - tr_y)  
        feet_y = max(bl_y, tr_y)
        raw_safe_y = feet_y + (box_h * safety_multiplier)

        MAX_FLOOR_Y = FRAME_HEIGHT - 5
        MIN_FLOOR_Y = 0
        raw_safe_y = max(MIN_FLOOR_Y, min(MAX_FLOOR_Y, raw_safe_y))

        smoothed_cx = (EMA_ALPHA * raw_cx) + ((1.0 - EMA_ALPHA) * self.last_cx)
        smoothed_safe_y = (EMA_ALPHA * raw_safe_y) + ((1.0 - EMA_ALPHA) * self.last_safe_y)

        self.last_cx = smoothed_cx
        self.last_safe_y = smoothed_safe_y

        src_point = np.array([[[smoothed_cx, smoothed_safe_y]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, H)

        pan_target = float(dst_point[0][0][0])
        tilt_target = float(dst_point[0][0][1])

        pan_target = max(0.0, min(180.0, pan_target))
        tilt_target = max(0.0, min(180.0, tilt_target))
        print(f"pan_target:{pan_target}, tilt_target:{tilt_target}")
        return pan_target, tilt_target

    def update_actuators(self, target_pan, target_tilt):
        """Safely writes pre-smoothed angles to PCA9685 over I2C."""
        try:
            # Angles are already smooth, write them directly
            self.kit.servo[PAN_CHANNEL].angle = target_pan
            self.kit.servo[TILT_CHANNEL].angle = target_tilt

            # Metric Tracking: Check if active displacement is occurring
            delta_pan = abs(target_pan - self.last_pan)
            delta_tilt = abs(target_tilt - self.last_tilt)

            # Threshold to consider "active motion" (e.g., > 0.5 degrees)
            if delta_pan > 0.5 or delta_tilt > 0.5:
                if self.motion_start_time is None:
                    self.motion_start_time = time.time()
            else:
                if self.motion_start_time is not None:
                    self.total_motion_duration += time.time() - self.motion_start_time
                    self.motion_start_time = None

            # Save angle state
            self.last_pan = target_pan
            self.last_tilt = target_tilt

        except OSError as e:
            # Catch I2C bus drops gracefully without crashing background loop
            logging.warning(f"I2C Write timeout/dropout detected: {e}. Re-trying next frame.")

    def run(self):
        logging.info("Starting edge tracking loop...")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Dropped frame detected from camera feed.")
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.flip(frame, 0)
                # 1. Processing & Orientation Correction
                input_tensor = self.preprocess_frame(frame)

                # 2. Inference
                try:
                    onnx_outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                    heatmaps = [onnx_outputs[0][0, 0], onnx_outputs[1][0, 0], onnx_outputs[1][0, 1], onnx_outputs[2][0, 0], onnx_outputs[2][0, 1]]
                    points, confidence = decode_image_to_full_scale(heatmaps)


                    if confidence > 0.3:
                        # Dummy placement: Replace with parsing logic from your CajuNet output matrix
                        bl, tr = points

                    else:
                        continue

                except Exception as e:
                    logging.error(f"Inference Failure: {e}")
                    continue  # Protect system loop from corrupt predictions

                # 3. Perspective Correction Mapping
                target_pan, target_tilt = self.calculate_safe_floor_angles(bl, tr)

                # 4. Kinematic Motion Control
                self.update_actuators(target_pan, target_tilt)

        except KeyboardInterrupt:
            logging.info("Shutdown sequence initiated by user.")
        finally:
            self.cleanup()

    def run_test_dataset(self, image_folder_path, loop_delay=1.0):

        logging.info("Starting offline dataset tracking loop using folder")
        dataset_path = Path(image_folder_path)
        image_paths = []
        images_paths = dataset_path.glob("*.png")

        image_paths = sorted([img for img in images_paths], key=lambda x:x.name) # Sorts chronologically or numerically by file name

        if not image_paths:
            logging.error("No valid images found in folder. Exiting loop.")
            return

        logging.info(f"Found {len(image_paths)} images to process. Energizing laser...")
        try:
            self.init_hardware()

            for idx, img_path in enumerate(image_paths):
                logging.info(f"\n[{idx + 1}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")


                input_tensor = self.preprocess_frame(img_path)

                # 2. Inference
                try:
                    onnx_outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                    heatmaps = [onnx_outputs[0][0, 0], onnx_outputs[1][0, 0], onnx_outputs[1][0, 1], onnx_outputs[2][0, 0], onnx_outputs[2][0, 1]]
                    points, confidence = decode_image_to_full_scale(heatmaps)


                    if confidence > 0.3:
                        # Dummy placement: Replace with parsing logic from your CajuNet output matrix
                        bl, tr = points

                    else:
                        continue

                except Exception as e:
                    logging.error(f"Inference Failure: {e}")
                    continue  # Protect system loop from corrupt predictions

                # 3. Perspective Correction Mapping
                target_pan, target_tilt = self.calculate_safe_floor_angles(bl, tr)

                # 4. Kinematic Motion Control
                self.update_actuators(target_pan, target_tilt)
                time.sleep(loop_delay)

        except KeyboardInterrupt:
            logging.info("Shutdown sequence initiated by user.")
            self.cleanup()
        finally:
            self.cleanup()

    def cleanup(self):
        """Graceful teardown to protect equipment."""
        if self.motion_start_time is not None:
            self.total_motion_duration += time.time() - self.motion_start_time

        logging.info(f"Total tracking active motion duration: {self.total_motion_duration:.2f} seconds.")

        try:
            # Turn off laser safely
            self.kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
            # Center servos
            self.kit.servo[PAN_CHANNEL].angle = 90
            self.kit.servo[TILT_CHANNEL].angle = 90
        except Exception:
            pass

        if self.cap:
            self.cap.release()
        logging.info("System safely disarmed.")




if __name__ == "__main__":
    tracker = EdgeTracker()
    dataset_path = '/home/razan236/Developer/cat_tracker/dataset/images'
    tracker.run_test_dataset(dataset_path, loop_delay=1.0)

