"""Camera capture module for Pi Camera Module 3."""

import threading
import time
import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

import config


class Camera:
    """Threaded camera capture."""

    def __init__(self, width=None, height=None):
        self.width = width or config.CAMERA_WIDTH
        self.height = height or config.CAMERA_HEIGHT
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None
        self._camera = None

    def start(self):
        if PICAMERA_AVAILABLE:
            self._camera = Picamera2()
            cam_config = self._camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                buffer_count=4
            )
            self._camera.configure(cam_config)
            self._camera.start()
            try:
                self._camera.set_controls({"AfMode": 2, "AfTrigger": 0})
                print("[Camera] Continuous autofocus enabled")
            except Exception as e:
                print(f"[Camera] Autofocus not available: {e}")
            time.sleep(2)
        else:
            self._camera = cv2.VideoCapture(0)
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Started at {self.width}x{self.height}")

    def _loop(self):
        while self.running:
            try:
                if PICAMERA_AVAILABLE:
                    frame = self._camera.capture_array()
                else:
                    ret, frame = self._camera.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                with self.lock:
                    self.frame = frame
            except Exception as e:
                print(f"[Camera] Error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        if PICAMERA_AVAILABLE and self._camera:
            self._camera.stop()
        elif self._camera:
            self._camera.release()
        print("[Camera] Stopped")
