import time
import sys
import logging
import os
import datetime
from pathlib import Path
import busio
import board
import cv2
import numpy as np
import onnxruntime as ort
import json
from adafruit_servokit import ServoKit
from map_servo_coordinates_to_image import H, pixel_to_servo
from prepare_images_for_onnx_model import preprocess_for_onnx, decode_image_to_full_scale

import toml

# --- NEW FIREBASE MODULE ENGINES ---
import firebase_admin
from firebase_admin import credentials, db, storage

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# PCA9685/Servo Channels
PAN_CHANNEL = 0
TILT_CHANNEL = 1
LASER_CHANNEL = 4

# Smoothing factor
EMA_ALPHA = 0.85

class EdgeTracker:
    def __init__(self, model_path="cat_centernet.onnx"):
        self.model_path = model_path
        self.kit = None
        self.ort_session = None
        self.cap = None

        # State tracking variables
        self.last_pan = 90.0
        self.last_tilt = 90.0
        self.total_motion_duration = 0.0
        self.motion_start_time = None
        self.last_cx = FRAME_WIDTH / 2.0
        self.last_safe_y = FRAME_HEIGHT / 2.0

        # --- NEW: CORE PLAYTIME AUTOMATION STATE MACHINE ---
        self.session_active = False
        self.session_start_time = None
        self.consecutive_no_target_frames = 0
        self.current_frame_cache = None  # Holds the initial snapshot frame
        
        # Initialize cloud network and local systems
        self.init_firebase()
        self.init_hardware()
        self.init_inference_engine()
        # self.init_camera() # Uncomment when running live video loop

    def init_firebase(self):
        """Extracts cloud database environments securely out of Streamlit secrets file."""
        try:
            # Point this path directly to where your webapp directory lives on your machine
            secrets_path = Path(__file__).parent / "webapp" / ".streamlit" / "secrets.toml"
            
            if not secrets_path.exists():
                logging.error(f"Could not locate secrets directory profile at: {secrets_path}")
                sys.exit(1)
                
            # Load and parse the TOML map keys
            secrets = toml.load(secrets_path)
            
            # Reconstruct the Firebase JSON dictionary object back from the string
            cred_dict = json.loads(secrets["FIREBASE_CREDENTIALS_JSON"])
            cred = credentials.Certificate(cred_dict)
            
            firebase_admin.initialize_app(cred, {
                'databaseURL': secrets["FIREBASE_DB_URL"],
                'storageBucket': secrets["FIREBASE_STORAGE_BUCKET"]
            })
            
            self.db_ref = db.reference("tracking_system")
            self.bucket = storage.bucket()
            logging.info("Firebase services successfully initialized via shared Streamlit configuration profiling.")
            
            # Reset active live nodes on startup
            self.db_ref.child("active_session").set({
                "is_playing": False,
                "current_duration_sec": 0
            })
        except Exception as e:
            logging.error(f"Critical Error initializing Firebase Cloud Services: {e}")
            sys.exit(1)

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

        raw_safe_y = max(0, min(FRAME_HEIGHT - 5, raw_safe_y))

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
        return pan_target, tilt_target

    def update_actuators(self, target_pan, target_tilt):
        """Safely writes pre-smoothed angles to PCA9685 over I2C."""
        try:
            self.kit.servo[PAN_CHANNEL].angle = target_pan
            self.kit.servo[TILT_CHANNEL].angle = target_tilt

            delta_pan = abs(target_pan - self.last_pan)
            delta_tilt = abs(target_tilt - self.last_tilt)

            if delta_pan > 0.5 or delta_tilt > 0.5:
                if self.motion_start_time is None:
                    self.motion_start_time = time.time()
            else:
                if self.motion_start_time is not None:
                    self.total_motion_duration += time.time() - self.motion_start_time
                    self.motion_start_time = None

            self.last_pan = target_pan
            self.last_tilt = target_tilt

        except OSError as e:
            logging.warning(f"I2C Write timeout/dropout detected: {e}. Re-trying next frame.")

    def manage_play_session(self, target_detected, frame_to_save):
        """Tracks the session state machine and syncs details to Firebase Cloud."""
        loop_now = time.time()
        
        if target_detected:
            self.consecutive_no_target_frames = 0
            
            # --- STARTING A FRESH SESSION ---
            if not self.session_active:
                logging.info("🐾 Caju detected in the frame workspace! Starting active cloud session tracker.")
                self.session_active = True
                self.session_start_time = loop_now
                self.current_frame_cache = frame_to_save.copy() # Cache the target snapshot frame
                
                # Instantly notify the live dashboard viewport
                self.db_ref.child("active_session").update({
                    "is_playing": True,
                    "current_duration_sec": 0
                })
            else:
                # Continuous Session Update
                current_duration = int(loop_now - self.session_start_time)
                self.db_ref.child("active_session/current_duration_sec").set(current_duration)
        else:
            self.consecutive_no_target_frames += 1
            
            if self.session_active:
                current_duration = int(loop_now - self.session_start_time)
                self.db_ref.child("active_session/current_duration_sec").set(current_duration)

        # --- WRAPPING UP AN ACTIVE PLAY SESSION ---
        # If cat is gone for ~15 sequential frames (adjust based on your loop execution latency)
        if self.session_active and self.consecutive_no_target_frames >= 15:
            final_duration = int(time.time() - self.session_start_time)
            self.session_active = False
            
            # Reset the Live dashboard nodes back to sleep parameters
            self.db_ref.child("active_session").set({
                "is_playing": False,
                "current_duration_sec": 0
            })
            
            # Only store valid sessions that lasted longer than a brief 3-second accidental trigger
            if final_duration > 3:
                session_id = f"session_{int(time.time())}"
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                local_temp_img = "cloud_upload_temp.jpg"
                cloud_storage_path = f"sessions/{session_id}.jpg"
                
                # A. Save cached snapshot frame locally
                if self.current_frame_cache is not None:
                    cv2.imwrite(local_temp_img, self.current_frame_cache)
                else:
                    cv2.imwrite(local_temp_img, frame_to_save)
                
                try:
                    # B. Stream binary payload directly into Firebase Cloud Storage Bucket
                    logging.info(f"📤 Uploading session frame image asset directly to storage path: {cloud_storage_path}")
                    blob = self.bucket.blob(cloud_storage_path)
                    blob.upload_from_filename(local_temp_img)
                    
                    # C. Append session item configuration array inside Realtime Database History
                    logging.info(f"💾 Saving historical log node inside Firebase Realtime Database.")
                    self.db_ref.child("sessions").child(session_id).set({
                        "timestamp": timestamp,
                        "duration_sec": final_duration,
                        "image_path": cloud_storage_path
                    })
                    logging.info(f"✅ Session {session_id} archived smoothly! Played for: {final_duration}s")
                except Exception as ex:
                    logging.error(f"Cloud write failure during archiving: {ex}")
                finally:
                    if os.path.exists(local_temp_img):
                        os.remove(local_temp_img)

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
                
                input_tensor = self.preprocess_frame(frame)
                target_detected = False
                bl, tr = (0, 0), (0, 0)

                try:
                    onnx_outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                    heatmaps = [onnx_outputs[0][0, 0], onnx_outputs[1][0, 0], onnx_outputs[1][0, 1], onnx_outputs[2][0, 0], onnx_outputs[2][0, 1]]
                    points, confidence = decode_image_to_full_scale(heatmaps)

                    if confidence > 0.3:
                        bl, tr = points
                        target_detected = True
                except Exception as e:
                    logging.error(f"Inference Failure: {e}")

                # Pass status variables through the session architecture engine
                self.manage_play_session(target_detected, frame)

                if target_detected:
                    target_pan, target_tilt = self.calculate_safe_floor_angles(bl, tr)
                    self.update_actuators(target_pan, target_tilt)

        except KeyboardInterrupt:
            logging.info("Shutdown sequence initiated by user.")
        finally:
            self.cleanup()

    def run_test_dataset(self, image_folder_path, loop_delay=1.0):
        logging.info("Starting offline dataset tracking loop using folder...")
        dataset_path = Path(image_folder_path)
        images_paths = dataset_path.glob("*.png")
        image_paths = sorted([img for img in images_paths], key=lambda x: x.name)

        if not image_paths:
            logging.error("No valid images found in folder. Exiting loop.")
            return

        logging.info(f"Found {len(image_paths)} images to process. Running offline stack diagnostics...")
        try:
            for idx, img_path in enumerate(image_paths):
                logging.info(f"[{idx + 1}/{len(image_paths)}] Reading Frame Matrix: {os.path.basename(img_path)}")
                
                # Load frame for processing and visualization conversion
                frame = cv2.imread(str(img_path))
                if frame is None: continue
                
                input_tensor = self.preprocess_frame(frame)
                target_detected = False
                bl, tr = (0, 0), (0, 0)

                try:
                    onnx_outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                    heatmaps = [onnx_outputs[0][0, 0], onnx_outputs[1][0, 0], onnx_outputs[1][0, 1], onnx_outputs[2][0, 0], onnx_outputs[2][0, 1]]
                    points, confidence = decode_image_to_full_scale(heatmaps)

                    if confidence > 0.3:
                        bl, tr = points
                        target_detected = True
                except Exception as e:
                    logging.error(f"Inference Failure: {e}")

                # Track session status from image sequences
                self.manage_play_session(target_detected, frame)

                if target_detected:
                    target_pan, target_tilt = self.calculate_safe_floor_angles(bl, tr)
                    self.update_actuators(target_pan, target_tilt)
                
                time.sleep(loop_delay)

        except KeyboardInterrupt:
            logging.info("Shutdown sequence initiated by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Graceful teardown to protect equipment."""
        if self.motion_start_time is not None:
            self.total_motion_duration += time.time() - self.motion_start_time

        logging.info(f"Total tracking active motion duration: {self.total_motion_duration:.2f} seconds.")

        try:
            self.kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
            self.kit.servo[PAN_CHANNEL].angle = 90
            self.kit.servo[TILT_CHANNEL].angle = 90
        except Exception:
            pass

        if self.cap:
            self.cap.release()
        logging.info("System safely disarmed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    tracker = EdgeTracker()
    dataset_path = '/home/razan236/Developer/cat_tracker/dataset/images'
    tracker.run_test_dataset(dataset_path, loop_delay=1.0)