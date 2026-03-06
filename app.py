#!/usr/bin/env python3
"""
Pokémon Card Grader — Main Application
=======================================
Runs the complete pipeline: camera → detection → identification → grading → web UI.

Usage:
    python3 app.py [--port 5000] [--width 1280] [--height 720]
                   [--skip-frames 3] [--demo]
"""

import argparse
import threading
import time
import sys
import cv2

import config
from detector import CardDetector
from identifier import CardIdentifier
from grader import CardGrader
from web_ui import WebUI


class Pipeline:
    """
    Main processing pipeline.
    Continuously captures frames, detects cards, identifies and grades them,
    and makes results available to the web UI.
    """

    def __init__(self, camera, frame_skip=None):
        self.camera = camera
        self.detector = CardDetector()
        self.identifier = CardIdentifier()
        self.grader = CardGrader()
        self.frame_skip = frame_skip or config.FRAME_SKIP

        # Shared state (thread-safe)
        self._lock = threading.Lock()
        self._display_frame = None
        self._results = {'cards': [], 'fps': 0.0}
        self._running = False

    def start(self):
        """Start the processing loop in a background thread."""
        self.camera.start()
        self._running = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
        print("[Pipeline] Processing started")

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        self.camera.stop()

    def get_display_frame(self):
        """Get the latest annotated frame for the video stream."""
        with self._lock:
            return self._display_frame

    def get_results(self):
        """Get the latest detection/grading results as a dict."""
        with self._lock:
            return self._results.copy()

    def _process_loop(self):
        """Main processing loop."""
        frame_count = 0
        fps_start = time.time()
        fps_frames = 0
        current_fps = 0.0

        # Cache the last results so we can draw them every frame
        last_detections = []
        last_card_results = []

        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_frames += 1

            # Calculate FPS every second
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            # Only run heavy processing every N frames
            if frame_count % self.frame_skip == 0:
                last_detections, last_card_results = self._analyze_frame(frame)

            # Draw overlays on every frame (using cached results)
            display = frame.copy()
            self.detector.draw_detections(display, last_detections, last_card_results)

            # Draw FPS counter
            cv2.putText(display, f"{current_fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Build JSON-serializable results
            cards_json = []
            for r in last_card_results:
                cards_json.append({
                    'name': r['name'],
                    'confidence': r['confidence'],
                    'grade': r['grade'],
                    'score': r['score'],
                    'defects': r['defects'],
                    'details': r['details'],
                })

            with self._lock:
                self._display_frame = display
                self._results = {
                    'cards': cards_json,
                    'fps': round(current_fps, 1),
                }

    def _analyze_frame(self, frame):
        """Run detection, identification, and grading on a single frame."""
        # Resize for faster processing
        proc_frame = cv2.resize(frame, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT))
        scale_x = frame.shape[1] / config.PROCESS_WIDTH
        scale_y = frame.shape[0] / config.PROCESS_HEIGHT

        # Detect cards
        detections = self.detector.detect(proc_frame)

        # Scale detection coordinates back to original frame size
        for det in detections:
            det['corners'] = det['corners'] * np.array([scale_x, scale_y])
            x, y, w, h = det['bbox']
            det['bbox'] = (int(x * scale_x), int(y * scale_y),
                           int(w * scale_x), int(h * scale_y))
            # Re-crop from full resolution frame for better grading
            det['cropped'] = self._crop_from_full(frame, det['corners'])

        card_results = []
        for det in detections:
            cropped = det['cropped']
            if cropped is None or cropped.size == 0:
                continue

            # Identify the card
            id_result = self.identifier.identify(cropped)

            # Grade the card's condition
            grade_result = self.grader.grade(cropped)

            card_results.append({
                'name': id_result['name'],
                'confidence': id_result['confidence'],
                'matches': id_result['matches'],
                'grade': grade_result['grade'],
                'score': grade_result['score'],
                'defects': grade_result['defects'],
                'details': grade_result['details'],
            })

        return detections, card_results

    def _crop_from_full(self, frame, corners):
        """Perspective-transform crop from the full-resolution frame."""
        try:
            corners = corners.astype(np.float32)
            output_w, output_h = 300, 420

            dst = np.array([
                [0, 0], [output_w - 1, 0],
                [output_w - 1, output_h - 1], [0, output_h - 1]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(corners, dst)
            warped = cv2.warpPerspective(frame, matrix, (output_w, output_h))
            return warped
        except Exception:
            return None


# Need numpy for the scaling
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Pokémon Card Grader')
    parser.add_argument('--port', type=int, default=config.FLASK_PORT,
                        help='Web UI port')
    parser.add_argument('--width', type=int, default=config.CAMERA_WIDTH,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=config.CAMERA_HEIGHT,
                        help='Camera height')
    parser.add_argument('--skip-frames', type=int, default=config.FRAME_SKIP,
                        help='Process every Nth frame')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode without a real camera')
    args = parser.parse_args()

    # Update config
    config.CAMERA_WIDTH = args.width
    config.CAMERA_HEIGHT = args.height
    config.FRAME_SKIP = args.skip_frames
    config.FLASK_PORT = args.port

    # Initialize camera
    if args.demo:
        from camera import DemoCamera
        camera = DemoCamera(args.width, args.height)
    else:
        from camera import Camera
        camera = Camera(args.width, args.height)

    # Build pipeline
    pipeline = Pipeline(camera, frame_skip=args.skip_frames)

    # Start processing
    pipeline.start()

    # Start web UI (blocking)
    ui = WebUI(pipeline)
    try:
        ui.run(port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
