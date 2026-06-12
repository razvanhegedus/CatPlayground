# CatPlayground — Autonomous Edge-Computing Cat Tracker

CatPlayground is an autonomous edge-computing system that tracks and interacts with indoor cats in real time without requiring continuous human supervision.

Designed specifically for **Caju**, the system combines **computer vision, embedded hardware control, and cloud telemetry** to create an interactive laser-play environment that keeps pets active while running entirely on local hardware.

Unlike cloud-based solutions, CatPlayground performs inference directly on a **Raspberry Pi 4B**, minimizing latency and maintaining responsive physical interaction.

---

## Features

### Real-Time Edge Inference

* Runs a custom **CenterNet-based neural network (CatCenterNet / CajuNet)** directly on Raspberry Pi CPU
* Optimized with **ONNX Runtime**
* Sustains approximately **10 FPS** for real-time tracking

### Automated Physical Actuation

* Converts **2D camera detections → 3D movement commands**
* Uses **homography-based coordinate mapping**
* Controls a **pan–tilt laser system** through a **PCA9685 PWM driver**

### Cloud Telemetry & Analytics

* Asynchronously uploads:

  * play session statistics
  * captured snapshots
  * historical interaction logs
* Uses **Firebase Realtime Database + Cloud Storage**
* Sends automatic email notifications after completed sessions

### Remote Monitoring Dashboard

* Built with **Streamlit**
* Monitor:

  * historical tracking activity
  * daily play volume
  * live session analytics

---

# System Architecture

```text
┌─────────────────────────────────────────────┐
│           Raspberry Pi Edge Layer           │
├─────────────────────────────────────────────┤
│ Pi Camera → Crop → ONNX Inference           │
│                ↓                            │
│      EMA Trajectory Smoothing               │
│                ↓                            │
│ Homography Coordinate Mapping               │
│                ↓                            │
│ PCA9685 → Servo Motors → Laser              │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│          Cloud Telemetry Layer              │
├─────────────────────────────────────────────┤
│ Firebase Database                           │
│ Cloud Storage                               │
│ Session Analytics                           │
│ Email Notification Service                  │
└─────────────────────────────────────────────┘
```

### Edge & Actuation Layer

A Raspberry Pi camera captures live video frames.

The image stream is:

1. Spatially cropped
2. Passed through the ONNX inference engine
3. Smoothed using an **Exponential Moving Average (EMA)** filter *(α = 0.85)*
4. Converted into mechanical actuation commands

### Cloud Telemetry Layer

Tracking sessions are asynchronously uploaded over Wi-Fi to Firebase while preserving uninterrupted local execution.

---

# Hardware Requirements

| Component          | Model                                           |
| ------------------ | ----------------------------------------------- |
| Microcomputer      | Raspberry Pi 4B                                 |
| Camera             | Pi Camera Module V2 / V3                        |
| Motor Controller   | PCA9685 16-Channel PWM Driver                   |
| Actuators          | 2× Micro Servos                                 |
| Interaction Module | 5V Laser Module + NPN Transistor (1kΩ resistor) |
| Power              | External 5V DC Supply                           |

> External power is recommended to isolate servo noise from Raspberry Pi logic rails.

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/hegedusrazvan/CatPlayground.git
cd CatPlayground
```

---

## 2. Model Training Dependencies

Install if retraining **CatCenterNet** or generating datasets.

```bash
pip install -r requirements-train.txt
```

`requirements-train.txt`

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy
opencv-python
fiftyone
groundingdino-py
```

---

## 3. Edge Deployment Dependencies

Install directly on Raspberry Pi.

```bash
pip install -r requirements-edge.txt
```

`requirements-edge.txt`

```txt
onnxruntime
opencv-python-headless
numpy
smbus2
firebase-admin
streamlit
```

---

# Running the System

### Start Real-Time Tracking

```bash
python run_tracker.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Model Performance

CatCenterNet adopts an **anchor-free center-point detection strategy** instead of traditional bounding box regression.

| Metric           | Result    |
| ---------------- | --------- |
| Runtime          | ~10 FPS   |
| Localization MAE | 2.30 px   |
| Input Resolution | 160 × 224 |

### Safety Mechanism

To reduce direct eye exposure:

* Target location is projected near the **lowest visible region of the detected cat**
* Position is shifted slightly forward using a configurable **safety multiplier**

---

# Future Work

* Multi-species tracking support
* Multi-class focal loss optimization
* Improved prediction head architecture
* Bidirectional manual override via Streamlit
* WebSocket-powered live laser targeting
* Interactive remote control interface

---

# License

MIT License
