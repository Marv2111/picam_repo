"""Flask web server with scan trigger, streaming, and collection."""

import time
import cv2
from flask import Flask, Response, render_template, jsonify, request
import config
from cardmarket import CardmarketLookup
from collection import CardCollection


class WebUI:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cardmarket = CardmarketLookup()
        self.collection = CardCollection()
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html',
                                   col_count=self.collection.get_count())

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._stream(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/scan', methods=['POST'])
        def scan():
            """Trigger a scan of whatever is in the guide frame."""
            result = self.pipeline.scan_guide_frame()
            return jsonify(result)

        @self.app.route('/results')
        def results():
            return jsonify(self.pipeline.get_results())

        @self.app.route('/lookup_price')
        def lookup_price():
            name = request.args.get('name', '')
            number = request.args.get('number', '')
            if not name:
                return jsonify({"found": False, "error": "No name"})
            return jsonify(self.cardmarket.lookup(name, number or None))

        @self.app.route('/save_card', methods=['POST'])
        def save_card():
            data = request.get_json()
            if not data or not data.get('name'):
                return jsonify({"success": False}), 400
            card_id = self.collection.save_card(data)
            return jsonify({"success": True, "id": card_id})

        @self.app.route('/collection')
        def collection_view():
            return jsonify({"cards": self.collection.get_all(),
                            "value": self.collection.get_total_value()})

        @self.app.route('/delete_card', methods=['POST'])
        def delete_card():
            data = request.get_json()
            if data and data.get('id'):
                self.collection.delete_card(data['id'])
            return jsonify({"success": True})

    def _stream(self):
        while True:
            frame = self.pipeline.get_display_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            ret, buf = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    def run(self, host=None, port=None):
        host = host or config.FLASK_HOST
        port = port or config.FLASK_PORT
        print(f"\n{'='*50}")
        print(f"  Pokemon Card Grader")
        print(f"  http://localhost:{port}")
        print(f"  Collection: {self.collection.get_count()} cards")
        print(f"{'='*50}\n")
        self.app.run(host=host, port=port, threaded=True, debug=False)
