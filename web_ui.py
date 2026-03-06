"""
Web UI module.
Serves a local web dashboard with live camera stream, detection overlays,
and card grading results using Flask.
"""

import json
import time
import threading
import cv2
from flask import Flask, Response, render_template, jsonify
import config


class WebUI:
    """Flask-based web dashboard for the card grader."""

    def __init__(self, pipeline):
        """
        Args:
            pipeline: reference to the main Pipeline object which provides
                      get_display_frame() and get_results()
        """
        self.pipeline = pipeline
        self.app = Flask(__name__,
                         template_folder='templates',
                         static_folder='static')
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            ref_count = self.pipeline.identifier.get_card_count()
            return render_template('index.html', ref_count=ref_count)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/results')
        def results():
            data = self.pipeline.get_results()
            return jsonify(data)

    def _generate_frames(self):
        """Generate MJPEG stream frames."""
        while True:
            frame = self.pipeline.get_display_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            ret, buffer = cv2.imencode('.jpg', frame,
                                        [cv2.IMWRITE_JPEG_QUALITY,
                                         config.JPEG_QUALITY])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

    def run(self, host=None, port=None):
        """Start the Flask server."""
        host = host or config.FLASK_HOST
        port = port or config.FLASK_PORT
        print(f"\n{'=' * 50}")
        print(f"  Pokemon Card Grader is running!")
        print(f"  Open http://localhost:{port} in your browser")
        print(f"{'=' * 50}\n")
        self.app.run(host=host, port=port, threaded=True, debug=False)
