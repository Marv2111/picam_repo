"""
Web UI module.
Serves a local web dashboard with live camera stream, detection overlays,
card grading results, Cardmarket price lookup, and card collection management.
"""

import time
import threading
import cv2
from flask import Flask, Response, render_template, jsonify, request
import config
from cardmarket import CardmarketLookup
from collection import CardCollection


class WebUI:
    """Flask-based web dashboard for the card grader."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cardmarket = CardmarketLookup()
        self.collection = CardCollection()
        self.app = Flask(__name__,
                         template_folder='templates',
                         static_folder='static')
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            ref_count = self.pipeline.identifier.get_card_count()
            col_count = self.collection.get_count()
            return render_template('index.html', ref_count=ref_count,
                                   col_count=col_count)

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

        @self.app.route('/lookup_price')
        def lookup_price():
            """Look up card price on Cardmarket (GET only)."""
            name = request.args.get('name', '')
            number = request.args.get('number', '')
            if not name:
                return jsonify({"found": False, "error": "No card name provided"})

            result = self.cardmarket.lookup(name, number or None)
            return jsonify(result)

        @self.app.route('/save_card', methods=['POST'])
        def save_card():
            """Save a card to the local collection."""
            data = request.get_json()
            if not data or not data.get('name'):
                return jsonify({"success": False, "error": "No card data"}), 400

            card_id = self.collection.save_card(data)
            return jsonify({"success": True, "id": card_id})

        @self.app.route('/collection')
        def collection_view():
            """Get all saved cards as JSON."""
            cards = self.collection.get_all()
            value = self.collection.get_total_value()
            return jsonify({"cards": cards, "value": value})

        @self.app.route('/delete_card', methods=['POST'])
        def delete_card():
            """Delete a card from the collection."""
            data = request.get_json()
            card_id = data.get('id') if data else None
            if not card_id:
                return jsonify({"success": False, "error": "No card ID"}), 400

            self.collection.delete_card(card_id)
            return jsonify({"success": True})

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
        col_count = self.collection.get_count()
        print(f"\n{'=' * 50}")
        print(f"  Pokémon Card Grader is running!")
        print(f"  Open http://localhost:{port} in your browser")
        print(f"  Collection: {col_count} cards saved")
        print(f"{'=' * 50}\n")
        self.app.run(host=host, port=port, threaded=True, debug=False)
