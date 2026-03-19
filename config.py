"""
Configuration constants for the Pokémon Card Grader system.
Adjust these values to tune detection and grading behavior.
"""

# --- Camera ---
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
PROCESS_WIDTH = 640       # Resolution used for processing (smaller = faster)
PROCESS_HEIGHT = 480

# --- Card Detection ---
CARD_ASPECT_RATIO = 0.716          # Standard Pokémon card: 63mm / 88mm
CARD_ASPECT_TOLERANCE = 0.12       # How far from ideal ratio is allowed
CARD_MIN_AREA_RATIO = 0.03         # Min card area as fraction of frame area
CARD_MAX_AREA_RATIO = 0.85         # Max card area as fraction of frame area
GAUSSIAN_BLUR_KERNEL = (5, 5)
CANNY_LOW = 30
CANNY_HIGH = 120
CONTOUR_APPROX_EPSILON = 0.02      # Contour approximation tolerance

# --- Card Identification (ORB) ---
ORB_FEATURES = 500                 # Number of ORB features to extract
MATCH_DISTANCE_THRESHOLD = 50      # Max Hamming distance for a "good" match
MIN_GOOD_MATCHES = 8               # Minimum good matches to declare identification
REFERENCE_DB_PATH = "reference_db.pkl"
REFERENCE_CARDS_DIR = "reference_cards"

# --- Condition Grading ---
# Edge whitening
EDGE_BORDER_WIDTH_FRAC = 0.04      # How far inward from card edge to check
EDGE_WHITE_THRESHOLD = 200         # Pixel value above this = "white"
EDGE_WHITE_SCORE_SCALE = 3.0       # Scaling factor for whitening score

# Scratches
SCRATCH_KERNEL_SIZE = 3
SCRATCH_THRESHOLD = 40             # High-pass response threshold
SCRATCH_SCORE_SCALE = 5.0

# Dirt
DIRT_DARK_THRESHOLD = 50           # Pixel value below this on light areas = dirt
DIRT_SCORE_SCALE = 4.0

# Bends
BEND_LINE_THRESHOLD = 50
BEND_MIN_LINE_LENGTH = 40
BEND_SCORE_SCALE = 4.0

# Centering
CENTER_TOLERANCE = 0.15            # Allowed deviation in border ratios

# Grade weights (must sum to 1.0)
WEIGHT_EDGE_WHITENING = 0.25
WEIGHT_SCRATCHES = 0.20
WEIGHT_DIRT = 0.15
WEIGHT_BENDS = 0.20
WEIGHT_CENTERING = 0.20

# Grade thresholds
GRADE_THRESHOLDS = {
    "Mint":      (0.00, 0.10),
    "Near Mint": (0.10, 0.25),
    "Excellent": (0.25, 0.45),
    "Good":      (0.45, 0.65),
    "Poor":      (0.65, 1.00),
}

# --- UI / Flask ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
JPEG_QUALITY = 80
FRAME_SKIP = 3                     # Process every Nth frame (1 = every frame)

# --- Colors (BGR for OpenCV) ---
COLOR_BOX = (0, 255, 0)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_GRADE_MINT = (0, 255, 0)
COLOR_GRADE_NM = (0, 200, 100)
COLOR_GRADE_EX = (0, 180, 255)
COLOR_GRADE_GOOD = (0, 100, 255)
COLOR_GRADE_POOR = (0, 0, 255)
