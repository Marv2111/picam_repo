"""
Camera capture module using Picamera2 for Raspberry Pi Camera Module 3.
Provides threaded frame capture for smooth streaming.
"""

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
    """Threaded camera capture from Pi Camera Module 3."""

    def __init__(self, width=None, height=None, use_picamera=True):
        self.width = width or config.CAMERA_WIDTH
        self.height = height or config.CAMERA_HEIGHT
        self.use_picamera = use_picamera and PICAMERA_AVAILABLE
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None
        self._camera = None

    def start(self):
        """Initialize and start the camera capture thread."""
        if self.use_picamera:
            self._init_picamera()
        else:
            self._init_usb_camera()

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Started ({'Picamera2' if self.use_picamera else 'USB/OpenCV'})"
              f" at {self.width}x{self.height}")

    def _init_picamera(self):
        """Initialize Picamera2."""
        self._camera = Picamera2()
        cam_config = self._camera.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            buffer_count=4
        )
        self._camera.configure(cam_config)
        self._camera.start()
        # Allow auto-exposure to settle
        time.sleep(2)

    def _init_usb_camera(self):
        """Fallback: Initialize a USB camera via OpenCV."""
        self._camera = cv2.VideoCapture(0)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._camera.isOpened():
            raise RuntimeError("Could not open USB camera")

    def _capture_loop(self):
        """Continuously capture frames in a background thread."""
        while self.running:
            try:
                if self.use_picamera:
                    frame = self._camera.capture_array()
                    # Picamera2 returns RGB; convert to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = self._camera.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                with self.lock:
                    self.frame = frame

            except Exception as e:
                print(f"[Camera] Capture error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """Return the most recent frame (BGR), or None if not available."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        """Stop the camera and release resources."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

        if self.use_picamera and self._camera is not None:
            self._camera.stop()
        elif self._camera is not None:
            self._camera.release()

        print("[Camera] Stopped")


class DemoCamera:
    """
    A fake camera for testing without hardware.
    Generates a synthetic frame with a colored rectangle (fake card).
    """

    def __init__(self, width=None, height=None):
        self.width = width or config.CAMERA_WIDTH
        self.height = height or config.CAMERA_HEIGHT

    def start(self):
        print("[DemoCamera] Running in demo mode (no real camera)")

    def get_frame(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)  # Dark gray background

        # Draw a fake card rectangle
        card_w, card_h = 200, 280
        x1 = (self.width - card_w) // 2
        y1 = (self.height - card_h) // 2
        cv2.rectangle(frame, (x1, y1), (x1 + card_w, y1 + card_h),
                       (200, 180, 50), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + card_w, y1 + card_h),
                       (255, 255, 255), 2)
        cv2.putText(frame, "DEMO CARD", (x1 + 30, y1 + card_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def stop(self):
        print("[DemoCamera] Stopped")
