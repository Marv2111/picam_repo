"""Configuration constants."""

# --- Camera ---
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# --- Guide Frame (percentage of frame) ---
# Card ratio: 63mm x 88mm = 0.716
GUIDE_WIDTH_PCT = 0.90    # Guide box width as % of frame width
GUIDE_ASPECT = 0.716      # width/height ratio

# --- Auto-detect ---
AUTO_DETECT_ENABLED = True
CARD_ASPECT_RATIO = 0.716
CARD_ASPECT_TOLERANCE = 0.12
CARD_MIN_AREA_RATIO = 0.05
CARD_MAX_AREA_RATIO = 0.80

# --- Grading weights (sum to 1.0) ---
WEIGHT_EDGE_WHITENING = 0.25
WEIGHT_SCRATCHES = 0.20
WEIGHT_DIRT = 0.15
WEIGHT_BENDS = 0.20
WEIGHT_CENTERING = 0.20

GRADE_THRESHOLDS = {
    "Mint":      (0.00, 0.10),
    "Near Mint": (0.10, 0.25),
    "Excellent": (0.25, 0.45),
    "Good":      (0.45, 0.65),
    "Poor":      (0.65, 1.00),
}

# --- Flask ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
JPEG_QUALITY = 85

# --- Colors (BGR) ---
COLOR_GUIDE = (0, 200, 255)       # Yellow guide frame
COLOR_GUIDE_ACTIVE = (0, 255, 0)  # Green when card detected
COLOR_BOX = (0, 255, 0)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
