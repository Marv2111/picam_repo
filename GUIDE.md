# Pokémon Card Grader — Raspberry Pi 5 Local Vision System

## Complete Implementation Guide

A fully offline computer vision system running on Raspberry Pi 5 that detects Pokémon trading cards from a live camera feed, identifies them, grades their physical condition, and displays results through a local web dashboard.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Raspberry Pi Setup](#2-raspberry-pi-setup)
3. [Project Structure](#3-project-structure)
4. [Card Detection](#4-card-detection)
5. [Card Identification](#5-card-identification)
6. [Condition Grading](#6-condition-grading)
7. [Web User Interface](#7-web-user-interface)
8. [Running the System](#8-running-the-system)
9. [Performance Optimization](#9-performance-optimization)
10. [Future Improvements](#10-future-improvements)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5                        │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Pi Camera│───▶│ Frame Capture│───▶│ Card Detector │  │
│  │ Module 3 │    │ (Picamera2)  │    │ (Contour/YOLO)│  │
│  └──────────┘    └──────────────┘    └──────┬────────┘  │
│                                             │           │
│                                    ┌────────▼────────┐  │
│                                    │  Card Cropper   │  │
│                                    │  & Preprocessor │  │
│                                    └────────┬────────┘  │
│                              ┌──────────────┼────────┐  │
│                              │              │        │  │
│                     ┌────────▼───┐  ┌───────▼──────┐ │  │
│                     │   Card ID  │  │  Condition   │ │  │
│                     │ (ORB Match │  │  Grader      │ │  │
│                     │  or TFLite)│  │  (OpenCV)    │ │  │
│                     └────────┬───┘  └───────┬──────┘ │  │
│                              └──────┬───────┘        │  │
│                              ┌──────▼───────┐        │  │
│                              │  Results      │        │  │
│                              │  Aggregator   │        │  │
│                              └──────┬───────┘        │  │
│                              ┌──────▼───────┐        │  │
│                              │  Flask Web UI │        │  │
│                              │  (port 5000)  │        │  │
│                              └──────────────┘        │  │
└─────────────────────────────────────────────────────────┘
         Browser: http://localhost:5000
```

### Pipeline Overview

| Stage | Technology | Purpose |
|-------|-----------|---------|
| Capture | Picamera2 | Grab frames from Pi Camera Module 3 |
| Detection | OpenCV contours or YOLOv8n | Find card rectangles in the frame |
| Cropping | OpenCV perspective transform | Extract and straighten card images |
| Identification | ORB feature matching or TFLite CNN | Match card against a reference database |
| Grading | OpenCV image analysis | Detect defects (scratches, dirt, whitening, centering) |
| UI | Flask + MJPEG stream | Display live feed with overlays and results |

---

## 2. Raspberry Pi Setup

### 2.1 Update the System

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 2.2 Enable the Camera

The Pi Camera Module 3 should work out of the box on Raspberry Pi OS (Bookworm 64-bit). Verify:

```bash
# Check that the camera is detected
libcamera-hello --list-cameras

# Quick test — shows a 5-second preview window
libcamera-hello -t 5000

# Take a test photo
libcamera-still -o test.jpg
```

If no camera is detected, check the ribbon cable connection and ensure it's in the correct CSI port.

### 2.3 Install System Dependencies

```bash
sudo apt install -y \
  python3-pip \
  python3-venv \
  python3-opencv \
  python3-picamera2 \
  python3-libcamera \
  python3-numpy \
  libopencv-dev \
  libatlas-base-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libhdf5-dev
```

### 2.4 Create a Python Virtual Environment

```bash
mkdir -p ~/pokemon-card-grader
cd ~/pokemon-card-grader

# Create venv with access to system packages (needed for picamera2)
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

### 2.5 Install Python Packages

```bash
pip install \
  flask \
  tflite-runtime \
  scikit-image \
  Pillow
```

> **Note:** We use `--system-site-packages` so the venv can access `picamera2` and `opencv` which are installed at the system level on Raspberry Pi OS.

### 2.6 Verify Everything Works

```bash
python3 -c "
import cv2; print(f'OpenCV: {cv2.__version__}')
from picamera2 import Picamera2; print('Picamera2: OK')
import numpy as np; print(f'NumPy: {np.__version__}')
from flask import Flask; print('Flask: OK')
print('All dependencies ready!')
"
```

---

## 3. Project Structure

```
pokemon-card-grader/
├── venv/                       # Python virtual environment
├── app.py                      # Main application entry point
├── config.py                   # Configuration constants
├── camera.py                   # Camera capture module
├── detector.py                 # Card detection (contour-based)
├── identifier.py               # Card identification (ORB matching)
├── grader.py                   # Condition grading engine
├── web_ui.py                   # Flask web server and streaming
├── templates/
│   └── index.html              # Web dashboard
├── static/
│   └── style.css               # Dashboard styles
├── reference_cards/            # Reference card images for matching
│   ├── pikachu_base_set.jpg
│   ├── charizard_base_set.jpg
│   └── ...
├── models/                     # (Optional) TFLite models
│   └── card_classifier.tflite
└── tools/
    └── build_reference_db.py   # Script to pre-compute ORB features
```

---

## 4. Card Detection

The detection stage finds rectangular card-shaped objects in each camera frame. We use two approaches — pick the one that works best for your setup.

### Approach A: Contour-Based Detection (No ML Required)

This works well when the card is on a contrasting background (e.g., dark desk surface).

**How it works:**

1. Convert frame to grayscale
2. Apply Gaussian blur to reduce noise
3. Use adaptive thresholding or Canny edge detection
4. Find contours
5. Filter for quadrilateral contours with an area matching card proportions
6. Apply perspective transform to get a flat, upright card image

Pokémon cards have a standard size of **63mm × 88mm** (ratio ~0.716). We use this ratio to filter false detections.

### Approach B: YOLOv8-Nano Object Detection

For more robust detection in varied conditions, you can export a YOLOv8n model to TFLite or ONNX format and run it on the Pi. This requires:

1. Training or fine-tuning YOLOv8n on a dataset of card images
2. Exporting to TFLite with INT8 quantization
3. Running inference via `tflite-runtime`

**For beginners, Approach A is recommended** because it requires no model training and works reliably for single cards on a clean background. The code below implements Approach A with an architecture that lets you swap in YOLO later.

---

## 5. Card Identification

### Strategy: ORB Feature Matching

We use **ORB (Oriented FAST and Rotated BRIEF)** feature descriptors, which are fast enough for real-time use on the Pi. The system:

1. **Offline (once):** Compute ORB features for every reference card image and save them to a database file.
2. **Runtime:** Compute ORB features for the detected card and match against the database using a brute-force Hamming distance matcher.
3. Pick the reference card with the most good matches.

### Building Your Reference Database

You need reference images of the cards you want to identify. Options for obtaining images:

- **Scan your own collection** using a flatbed scanner at 300 DPI
- **Download images** from sites like pokemontcg.io (they offer an API with card images) — download once while you have internet, then use offline
- **Photograph cards** under consistent lighting using the Pi Camera itself

Place all reference images in the `reference_cards/` directory, named descriptively:

```
reference_cards/
├── pikachu_base_set_58.jpg
├── charizard_base_set_4.jpg
├── mewtwo_base_set_10.jpg
├── blastoise_base_set_2.jpg
└── ...
```

The more reference cards you add, the more cards the system can identify. Start with 10–20 cards to test, then expand.

---

## 6. Condition Grading

The grading engine analyzes the cropped card image for five defect categories:

| Defect | Detection Method | What It Looks For |
|--------|-----------------|-------------------|
| Edge Whitening | Color analysis in border region | White/light pixels along edges where color has worn |
| Scratches | High-pass filter + line detection | Thin bright lines across the card surface |
| Dirt/Stains | Color outlier detection | Dark spots or discoloration that don't match the card |
| Bends/Creases | Edge + gradient analysis | Sharp straight lines with shadow, indicating a fold |
| Centering | Border width measurement | Uneven borders around the card artwork |

### Grading Scale

Each defect category produces a score from 0.0 (perfect) to 1.0 (severe). These are combined into an overall grade:

| Grade | Score Range | Description |
|-------|-------------|-------------|
| Mint | 0.00 – 0.10 | Virtually perfect |
| Near Mint | 0.10 – 0.25 | Minor imperfections only visible on close inspection |
| Excellent | 0.25 – 0.45 | Light wear, minor whitening |
| Good | 0.45 – 0.65 | Moderate wear, visible defects |
| Poor | 0.65 – 1.00 | Heavy wear, major defects |

---

## 7. Web User Interface

The dashboard runs as a Flask app on `http://localhost:5000` and shows:

- **Left panel:** Live MJPEG camera stream with bounding box overlays
- **Right panel:** Card information — name, confidence, grade, defect breakdown

The stream is served via the `/video_feed` endpoint using `multipart/x-mixed-replace` so it works in any browser with no JavaScript required for the video portion. Card results are polled via a lightweight JSON endpoint.

---

## 8. Running the System

### Quick Start

```bash
cd ~/pokemon-card-grader
source venv/bin/activate

# Step 1: Build the reference database (run once, or when you add cards)
python3 tools/build_reference_db.py

# Step 2: Launch the application
python3 app.py
```

Then open a browser on the Pi and go to **http://localhost:5000**.

### Command-Line Options

```bash
# Run with custom resolution
python3 app.py --width 1280 --height 720

# Run with higher frame processing rate
python3 app.py --skip-frames 1

# Run on a different port
python3 app.py --port 8080
```

---

## 9. Performance Optimization

### Tips for Smooth Operation on Raspberry Pi 5

| Technique | Impact | How |
|-----------|--------|-----|
| Frame skipping | High | Only run detection/grading every N frames |
| Reduced resolution | High | Use 640×480 for processing, full res for display |
| ROI processing | Medium | Only process the region where a card was last seen |
| ORB feature limit | Medium | Cap at 500 features per image |
| NumPy vectorization | Medium | Avoid Python loops for pixel operations |
| Process separation | Medium | Run camera, detection, and UI in separate threads |
| INT8 quantization | High | If using TFLite, quantize models to INT8 |

### Raspberry Pi 5 Specific Advantages

The Pi 5 has a significantly faster CPU (Cortex-A76) compared to Pi 4. Expected performance:

- Contour detection: ~15–25 FPS at 640×480
- ORB matching (against 100 reference cards): ~5–10 FPS
- Full pipeline with grading: ~3–8 FPS (depending on frame skip settings)

### Memory Management

```bash
# Monitor resource usage while running
htop

# If running headless (no desktop), free ~200MB RAM:
sudo systemctl set-default multi-user.target
sudo reboot
# Access the web UI from another device on the same network
```

---

## 10. Future Improvements

### PSA-Style 1–10 Grading

Map the current 5-tier grade to a numeric 1–10 scale by using finer thresholds and weighting sub-grades differently. PSA emphasizes centering and corners heavily — adjust weights accordingly.

### Better Identification with a Trained CNN

1. Download a full Pokémon card dataset (pokemontcg.io has 10,000+ card images)
2. Train a MobileNetV3-Small classifier using TensorFlow
3. Export to TFLite with INT8 quantization
4. Replace the ORB matcher with the CNN — expect much higher accuracy

### Custom YOLOv8 Card Detector

1. Label ~500 images of cards in various positions using Roboflow or CVAT
2. Train YOLOv8n: `yolo train model=yolov8n.pt data=cards.yaml epochs=50 imgsz=640`
3. Export: `yolo export model=best.pt format=tflite int8=True`
4. Drop the `.tflite` file into the `models/` directory

### Multi-Card Detection

The contour-based approach already supports multiple cards. To display results for several cards simultaneously, extend the results panel to show a list and assign each detection a tracking ID.

### Improved Defect Detection

- Use a U-Net segmentation model trained on card defect masks
- Add scratch severity classification (light/medium/deep)
- Use stereo or structured-light imaging for bend detection

---

## Appendix: Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not detected | Check ribbon cable, ensure correct CSI port, run `libcamera-hello --list-cameras` |
| Low FPS | Reduce resolution, increase frame skip, close other applications |
| Cards not detected | Improve lighting, use a contrasting background, adjust threshold parameters |
| Wrong card identified | Add more reference images, ensure reference images are high quality |
| Import errors | Make sure venv is activated and was created with `--system-site-packages` |
| Flask won't start | Check port isn't already in use: `lsof -i :5000` |
