#!/usr/bin/env python3
"""
Pokémon Card Grader — Main Application
Guide-frame scan (primary) + auto-detect (bonus).
"""

import argparse
import threading
import time
import cv2
import numpy as np

import config
from camera import Camera
from detector import CardDetector
from identifier import CardIdentifier
from grader import CardGrader
from web_ui import WebUI


class Pipeline:
    """Processing pipeline with guide-frame and auto-detect."""

    def __init__(self, camera):
        self.camera = camera
        self.detector = CardDetector()
        self.identifier = CardIdentifier()
        self.grader = CardGrader()

        self._lock = threading.Lock()
        self._display_frame = None
        self._scan_result = None        # Latest scan result
        self._auto_result = None        # Latest auto-detect result
        self._auto_detection = None     # Auto-detect box for overlay
        self._running = False
        self._frame_count = 0

    def start(self):
        self.camera.start()
        self._running = True
        threading.Thread(target=self._display_loop, daemon=True).start()
        print("[Pipeline] Ready — use the Scan button or auto-detect")

    def stop(self):
        self._running = False
        self.camera.stop()

    def get_display_frame(self):
        with self._lock:
            return self._display_frame

    def get_results(self):
        """Return current state for the UI."""
        with self._lock:
            return {
                "scan": self._scan_result,
                "auto": self._auto_result,
            }

    def scan_guide_frame(self):
        """
        Triggered by the Scan button.
        Crops the guide frame region, runs OCR + grading.
        Returns the result dict.
        """
        frame = self.camera.get_frame()
        if frame is None:
            return {"error": "No camera frame"}

        # Crop the guide region
        card_img = self.detector.crop_guide_region(frame)

        # Step 1: Identify (OCR reads name + number)
        id_result = self.identifier.identify(card_img)

        # Step 2: Grade condition
        grade_result = self.grader.grade(card_img)

        result = {
            "name": id_result["name"],
            "confidence": id_result["confidence"],
            "card_number": id_result["card_number"],
            "raw_ocr": id_result["raw_ocr"],
            "grade": grade_result["grade"],
            "score": grade_result["score"],
            "defects": grade_result["defects"],
            "details": grade_result["details"],
        }

        with self._lock:
            self._scan_result = result

        return result

    def _display_loop(self):
        """Continuously update the display frame with overlays."""
        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            display = frame.copy()
            self._frame_count += 1

            # Run auto-detect every 10 frames (low CPU cost)
            auto_det = None
            if config.AUTO_DETECT_ENABLED and self._frame_count % 10 == 0:
                try:
                    # Run on smaller image for speed
                    small = cv2.resize(frame, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT))
                    det = self.detector.auto_detect(small)

                    if det is not None:
                        # Scale corners back to full resolution
                        sx = frame.shape[1] / config.PROCESS_WIDTH
                        sy = frame.shape[0] / config.PROCESS_HEIGHT
                        scaled_corners = (det['corners'] * np.array([sx, sy], dtype=np.float32)).astype(np.float32)
                        det['corners'] = scaled_corners
                        x, y, w, h = det['bbox']
                        det['bbox'] = (int(x*sx), int(y*sy), int(w*sx), int(h*sy))
                        det['cropped'] = self.detector._perspective_transform(frame, scaled_corners)

                        # Quick OCR on auto-detected card
                        auto_id = self.identifier.identify(det['cropped'])
                        auto_det = det

                        with self._lock:
                            self._auto_detection = det
                            self._auto_result = {
                                "name": auto_id["name"],
                                "confidence": auto_id["confidence"],
                                "card_number": auto_id["card_number"],
                            }
                except Exception as e:
                    pass  # Auto-detect is a bonus, don't crash on errors

            # Clear stale auto-detect periodically
            if config.AUTO_DETECT_ENABLED and self._frame_count % 30 == 0 and auto_det is None:
                with self._lock:
                    self._auto_detection = None
                    self._auto_result = None

            # Draw auto-detect box if we have one
            with self._lock:
                ad = self._auto_detection
                ar = self._auto_result
            if ad is not None:
                self.detector.draw_auto_detection(display, ad, ar)

            # Draw guide frame (always visible)
            has_auto = ad is not None
            self.detector.draw_guide(display, active=has_auto)

            with self._lock:
                self._display_frame = display

            time.sleep(0.01)  # ~30fps display


def main():
    parser = argparse.ArgumentParser(description='Pokémon Card Grader')
    parser.add_argument('--port', type=int, default=config.FLASK_PORT)
    parser.add_argument('--width', type=int, default=config.CAMERA_WIDTH)
    parser.add_argument('--height', type=int, default=config.CAMERA_HEIGHT)
    parser.add_argument('--no-auto', action='store_true',
                        help='Disable auto-detection')
    args = parser.parse_args()

    config.CAMERA_WIDTH = args.width
    config.CAMERA_HEIGHT = args.height
    config.FLASK_PORT = args.port
    if args.no_auto:
        config.AUTO_DETECT_ENABLED = False

    camera = Camera(args.width, args.height)
    pipeline = Pipeline(camera)
    pipeline.start()

    ui = WebUI(pipeline)
    try:
        ui.run(port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
