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
from prepare_images_for_onnx_model import preprocess_image_array, decode_image_to_full_scale
import toml

# --- NEW IMPORTS FOR EMAIL ---
import smtplib
from email.message import EmailMessage
import threading

# --- FIREBASE MODULE ENGINES ---
import firebase_admin
from firebase_admin import credentials, db, storage

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# PCA9685/Servo Channels
PAN_CHANNEL = 0
TILT_CHANNEL = 2
LASER_CHANNEL = 4

# Smoothing factor
EMA_ALPHA = 0.35

class EdgeTracker:
    def __init__(self, model_path="cat_centernet.onnx", confidence_threshold=0.3, min_box_size=10, upload_every_n_frames=2):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_box_size = min_box_size  
        self.upload_every_n_frames = upload_every_n_frames  
        self.frame_counter = 0

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

        # --- PLAYTIME AUTOMATION STATE MACHINE ---
        self.session_active = False
        self.session_start_time = None
        self.consecutive_no_target_frames = 0
        self.current_frame_cache = None  
        
        self.highest_session_confidence = 0.0 
        self.largest_session_box_area = 0.0
        
        # Initialize cloud network, local systems, and notifications
        self.init_firebase_and_secrets()
        self.init_hardware()
        self.init_inference_engine()

    def init_firebase_and_secrets(self):
        try:
            secrets_path = Path(__file__).parent / "secrets.toml"
            if not secrets_path.exists():
                logging.error(f"Could not locate secrets directory profile at: {secrets_path}")
                sys.exit(1)
                
            secrets = toml.load(secrets_path)
            
            # Load Firebase
            cred_dict = json.loads(secrets["FIREBASE_CREDENTIALS_JSON"])
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': secrets["FIREBASE_DB_URL"],
                'storageBucket': secrets["FIREBASE_STORAGE_BUCKET"]
            })
            self.db_ref = db.reference("/")
            self.bucket = storage.bucket()
            
            self.db_ref.child("active_session").set({
                "is_playing": False,
                "current_duration_sec": 0
            })
            logging.info("Firebase services successfully initialized.")

            # Load Email Config
            self.email_sender = secrets.get("EMAIL_SENDER")
            self.email_password = secrets.get("EMAIL_PASSWORD")
            self.email_receiver = secrets.get("EMAIL_RECEIVER")
            if self.email_sender and self.email_password:
                logging.info("Email notification credentials loaded successfully.")
            else:
                logging.warning("Email credentials missing from secrets.toml. Notifications disabled.")

        except Exception as e:
            logging.error(f"Critical Error initializing Cloud Services: {e}")
            sys.exit(1)

    def init_hardware(self):
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            kit = ServoKit(channels=16, i2c=i2c)
            self.kit = kit
            self.kit.servo[PAN_CHANNEL].set_pulse_width_range(500, 2500)
            self.kit.servo[TILT_CHANNEL].set_pulse_width_range(500, 2500)
            self.kit.servo[PAN_CHANNEL].angle = self.last_pan
            self.kit.servo[TILT_CHANNEL].angle = self.last_tilt
            self.kit._pca.channels[LASER_CHANNEL].duty_cycle = 0x0FFF
            logging.info("Hardware stack (PCA9685) initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to communicate with PCA9685 over I2C: {e}")
            sys.exit(1)

    def init_inference_engine(self):
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2  
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.ort_session = ort.InferenceSession(self.model_path, opts)
            self.input_name = self.ort_session.get_inputs()[0].name
            logging.info(f"ONNX Model '{self.model_path}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ONNX model: {e}")
            sys.exit(1)

    def preprocess_frame(self, frame, is_live_stream=True):
        return preprocess_image_array(frame, downscale_factor=4, is_live_stream=is_live_stream)

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
        return max(0.0, min(180.0, pan_target)), max(0.0, min(180.0, tilt_target))

    def update_actuators(self, target_pan, target_tilt):
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
            logging.warning(f"I2C Write timeout/dropout detected: {e}.")

    def stream_live_frame_to_webapp(self, frame, latency_ms, fps, confidence, bl=None, tr=None):
        self.frame_counter += 1
        if self.frame_counter % self.upload_every_n_frames != 0:
            return 
            
        viz_frame = frame.copy()
        
        if confidence >= self.confidence_threshold and bl is not None and tr is not None:
            cv2.rectangle(viz_frame, (int(bl[0]), int(bl[1])), (int(tr[0]), int(tr[1])), (0, 255, 204), 2)
            cv2.putText(viz_frame, f"Cat: {confidence*100:.1f}%", (int(bl[0]), int(bl[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 204), 2)

        telemetry_text = f"Edge Core Info: {latency_ms:.1f}ms ({fps:.1f} FPS)"
        cv2.rectangle(viz_frame, (10, 10), (460, 45), (15, 17, 23), -1)
        cv2.putText(viz_frame, telemetry_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        temp_stream_name = "live_stream_temp.png"
        cv2.imwrite(temp_stream_name, viz_frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]) 
        
        try:
            blob = self.bucket.blob("live_monitor/feed.png")
            blob.upload_from_filename(temp_stream_name)
            self.db_ref.child("live_telemetry").set({
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "confidence": round(float(confidence), 2),
                "last_update": datetime.datetime.now().strftime("%H:%M:%S")
            })
        except Exception as e:
            logging.warning(f"Failed to sync live viewport frame to storage backend: {e}")
        finally:
            if os.path.exists(temp_stream_name):
                os.remove(temp_stream_name)

    # --- NEW: Asynchronous Email Sender ---
    def send_email_alert(self, session_id, duration):
        if not self.email_sender or not self.email_receiver:
            return

        def send_task():
            try:
                msg = EmailMessage()
                msg.set_content(f"Caju just finished playing!\n\nSession ID: {session_id}\nDuration: {duration} seconds.\nCheck your dashboard for the new image!")
                msg['Subject'] = f"🐾 Caju Play Session Completed!"
                msg['From'] = self.email_sender
                msg['To'] = self.email_receiver

                # Connects to Gmail's SMTP server (Change smtp.gmail.com if using Yahoo/Outlook)
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
                server.quit()
                logging.info(f"📧 Email alert sent successfully for {session_id}.")
            except Exception as e:
                logging.error(f"📧 Failed to send email alert: {e}")

        # Start the email process in the background so the robot doesn't freeze
        email_thread = threading.Thread(target=send_task)
        email_thread.start()

    def manage_play_session(self, target_detected, frame_to_save, confidence=0.0, bl=None, tr=None):
        loop_now = time.time()
        
        if target_detected:
            current_area = 0.0
            if bl is not None and tr is not None:
                current_area = abs(tr[0] - bl[0]) * abs(tr[1] - bl[1])

            self.consecutive_no_target_frames = 0
            if not self.session_active:
                logging.info("🐾 Caju detected in workspace! Activating active cloud session trackers.")
                self.session_active = True
                self.session_start_time = loop_now
                self.current_frame_cache = frame_to_save.copy()
                self.highest_session_confidence = confidence 
                self.largest_session_box_area = current_area
                self.db_ref.child("active_session").update({"is_playing": True, "current_duration_sec": 0})
            else:
                current_duration = int(loop_now - self.session_start_time)
                self.db_ref.child("active_session/current_duration_sec").set(current_duration)
                
                if confidence > 0.9 and current_area > self.largest_session_box_area:
                    self.current_frame_cache = frame_to_save.copy()
                    self.largest_session_box_area = current_area
                    self.highest_session_confidence = confidence
                    logging.info(f"📸 Better frame captured! Confidence: {confidence:.2f}, Area: {current_area:.0f}")
        else:
            self.consecutive_no_target_frames += 1
            if self.session_active:
                current_duration = int(loop_now - self.session_start_time)
                self.db_ref.child("active_session/current_duration_sec").set(current_duration)

        if self.session_active and self.consecutive_no_target_frames >= 15:
            final_duration = int(time.time() - self.session_start_time)
            self.session_active = False
            self.db_ref.child("active_session").set({"is_playing": False, "current_duration_sec": 0})
            
            if final_duration >= 0: 
                last_session_query = self.db_ref.child("sessions").order_by_key().limit_to_last(1).get()
                next_id_num = 1
                
                if last_session_query:
                    last_key = list(last_session_query.keys())[0]
                    try:
                        next_id_num = int(last_key.split("_")[1]) + 1
                    except (IndexError, ValueError):
                        logging.warning(f"Could not parse sequential ID from {last_key}. Defaulting to 1.")
                
                session_id = f"session_{next_id_num:03d}"
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                local_temp_img = "cloud_upload_temp.png"
                cloud_storage_path = f"sessions/{session_id}.png"
                
                cv2.imwrite(local_temp_img, self.current_frame_cache if self.current_frame_cache is not None else frame_to_save)
                
                try:
                    blob = self.bucket.blob(cloud_storage_path)
                    blob.upload_from_filename(local_temp_img)
                    self.db_ref.child("sessions").child(session_id).set({
                        "timestamp": timestamp, "duration_sec": final_duration, "image_path": cloud_storage_path
                    })
                    logging.info(f"✅ Session {session_id} archived! Duration: {final_duration}s")
                    
                    # --- NEW: Trigger the background email alert ---
                    self.send_email_alert(session_id, final_duration)

                except Exception as ex:
                    logging.error(f"Cloud write failure: {ex}")
                finally:
                    if os.path.exists(local_temp_img): os.remove(local_temp_img)

    def run_test_dataset(self, image_folder_path, target_fps=30.0):
        logging.info(f"Starting real-time stream simulation to web app at {target_fps} FPS...")
        dataset_path = Path(image_folder_path)
        image_paths = sorted(list(dataset_path.glob("*.png")), key=lambda x: x.name)

        if not image_paths:
            logging.error("No valid images found in folder.")
            return

        frame_interval = 1.0 / target_fps
        
        try:
            for idx, img_path in enumerate(image_paths):
                frame_start_time = time.perf_counter()
                frame = cv2.imread(str(img_path))
                if frame is None: continue
                
                input_tensor = self.preprocess_frame(frame, is_live_stream=False)
                target_detected = False
                bl, tr = None, None
                confidence = 0.0 

                try:
                    onnx_outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                    heatmaps = [onnx_outputs[0][0, 0], onnx_outputs[1][0, 0], onnx_outputs[1][0, 1], onnx_outputs[2][0, 0], onnx_outputs[2][0, 1]]
                    points, confidence = decode_image_to_full_scale(heatmaps)

                    if confidence >= self.confidence_threshold:
                        bl, tr = points
                        box_w = abs(tr[0] - bl[0])
                        box_h = abs(tr[1] - bl[1])
                        if box_w > self.min_box_size and box_h > self.min_box_size:
                            target_detected = True
                except Exception as e:
                    logging.error(f"Inference Failure: {e}")
                    confidence = 0.0

                self.manage_play_session(target_detected, frame, confidence, bl, tr)

                if target_detected:
                    target_pan, target_tilt = self.calculate_safe_floor_angles(bl, tr)
                    self.update_actuators(target_pan, target_tilt)
                
                elapsed = time.perf_counter() - frame_start_time
                latency_ms = elapsed * 1000.0
                current_fps = 1.0 / elapsed if elapsed > 0 else target_fps

                self.stream_live_frame_to_webapp(frame, latency_ms, current_fps, confidence, bl, tr)

                time_to_wait = frame_interval - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

            if self.session_active:
                logging.info("Dataset ended. Forcing save of final active session...")
                self.consecutive_no_target_frames = 15 
                fallback = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                self.manage_play_session(False, self.current_frame_cache if self.current_frame_cache is not None else fallback, 0.0, None, None)

        except KeyboardInterrupt:
            logging.info("Shutdown sequence initiated by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("Initiating system cleanup sequence...")
        
        if self.session_active:
            logging.info("Active session detected during shutdown. Forcing save...")
            self.consecutive_no_target_frames = 15 
            fallback_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) 
            self.manage_play_session(False, self.current_frame_cache if self.current_frame_cache is not None else fallback_frame, 0.0, None, None)
        
        try:
            self.db_ref.child("active_session").set({
                "is_playing": False,
                "current_duration_sec": 0
            })
            logging.info("Firebase active_session state reset to False.")
        except Exception as e:
            logging.error(f"Failed to reset Firebase state during cleanup: {e}")

        if self.motion_start_time is not None:
            self.total_motion_duration += time.time() - self.motion_start_time
        logging.info(f"Total active motion: {self.total_motion_duration:.2f}s")
        
        try:
            if self.kit is not None:
                self.kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
                self.kit.servo[PAN_CHANNEL].angle = 90
                self.kit.servo[TILT_CHANNEL].angle = 90
        except Exception: 
            pass
            
        if self.cap: 
            self.cap.release()
            
        logging.info("System disarmed safely.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    tracker = EdgeTracker(upload_every_n_frames=3) 
    dataset_path = '/home/razan236/Developer/cat_tracker/dataset/presentation_example'
    tracker.run_test_dataset(dataset_path, target_fps=30.0)