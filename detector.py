"""
Robust card detection module.
Finds rectangular card-shaped objects in a camera frame using MULTIPLE
detection strategies to work on any background:

1. Multi-threshold edge detection (light on dark AND dark on light)
2. Adaptive thresholding at multiple block sizes
3. Color channel separation (catches colored card borders)
4. Saturation-based detection (card surfaces differ from backgrounds)

All strategies produce candidate quadrilaterals which are filtered
for card-like aspect ratios and deduplicated.
"""

import cv2
import numpy as np
import config


class CardDetector:
    """Detect rectangular cards in a frame using multi-strategy analysis."""

    def __init__(self):
        self.min_area = None
        self.max_area = None

    def detect(self, frame):
        """
        Find cards in the given BGR frame using multiple strategies.

        Returns:
            list of dict, each containing:
                - 'corners': 4 corner points in the original frame
                - 'cropped': perspective-corrected card image (BGR)
                - 'bbox': (x, y, w, h) bounding rectangle
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        self.min_area = frame_area * config.CARD_MIN_AREA_RATIO
        self.max_area = frame_area * config.CARD_MAX_AREA_RATIO

        # Collect candidates from all strategies
        all_candidates = []

        # Strategy 1: Classic Canny at multiple thresholds
        all_candidates += self._detect_canny(frame, low=20, high=80)
        all_candidates += self._detect_canny(frame, low=40, high=130)
        all_candidates += self._detect_canny(frame, low=80, high=200)

        # Strategy 2: Adaptive thresholding (works when contrast is uneven)
        all_candidates += self._detect_adaptive_threshold(frame)

        # Strategy 3: Color channel separation (catches colored borders)
        all_candidates += self._detect_color_channels(frame)

        # Strategy 4: Saturation/brightness based
        all_candidates += self._detect_hsv(frame)

        # Strategy 5: Otsu on grayscale
        all_candidates += self._detect_otsu(frame)

        if not all_candidates:
            return []

        # Deduplicate overlapping detections
        cards = self._deduplicate(all_candidates, frame)

        return cards

    def _find_quads_from_edges(self, edges, frame):
        """Common logic: find card-like quadrilaterals from an edge map."""
        # Dilate to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Also try closing operation
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

        candidates = []
        for edge_img in [dilated, closed]:
            contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                result = self._process_contour(cnt, frame)
                if result is not None:
                    candidates.append(result)

            # Also try RETR_LIST to catch nested contours
            contours2, _ = cv2.findContours(edge_img, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours2:
                result = self._process_contour(cnt, frame)
                if result is not None:
                    candidates.append(result)

        return candidates

    # ------------------------------------------------------------------
    # Strategy 1: Canny edge detection
    # ------------------------------------------------------------------
    def _detect_canny(self, frame, low=30, high=120):
        """Canny edges at given thresholds."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low, high)
        return self._find_quads_from_edges(edges, frame)

    # ------------------------------------------------------------------
    # Strategy 2: Adaptive threshold
    # ------------------------------------------------------------------
    def _detect_adaptive_threshold(self, frame):
        """Adaptive thresholding at multiple block sizes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        candidates = []
        for block_size in [11, 21, 31]:
            # Normal
            thresh = cv2.adaptiveThreshold(blurred, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, block_size, 5)
            edges = cv2.Canny(thresh, 50, 150)
            candidates += self._find_quads_from_edges(edges, frame)

            # Inverted
            inv = cv2.bitwise_not(thresh)
            edges_inv = cv2.Canny(inv, 50, 150)
            candidates += self._find_quads_from_edges(edges_inv, frame)

        return candidates

    # ------------------------------------------------------------------
    # Strategy 3: Individual color channels
    # ------------------------------------------------------------------
    def _detect_color_channels(self, frame):
        """Run edge detection on individual BGR channels."""
        candidates = []
        for i in range(3):
            channel = frame[:, :, i]
            blurred = cv2.GaussianBlur(channel, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            candidates += self._find_quads_from_edges(edges, frame)
        return candidates

    # ------------------------------------------------------------------
    # Strategy 4: HSV saturation/value
    # ------------------------------------------------------------------
    def _detect_hsv(self, frame):
        """Use HSV color space — cards often have more saturation than backgrounds."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)

        candidates = []

        # Saturation-based edges
        sat_blur = cv2.GaussianBlur(sat, (5, 5), 0)
        edges_sat = cv2.Canny(sat_blur, 30, 100)
        candidates += self._find_quads_from_edges(edges_sat, frame)

        # Value-based edges
        val_blur = cv2.GaussianBlur(val, (5, 5), 0)
        edges_val = cv2.Canny(val_blur, 30, 100)
        candidates += self._find_quads_from_edges(edges_val, frame)

        return candidates

    # ------------------------------------------------------------------
    # Strategy 5: Otsu thresholding
    # ------------------------------------------------------------------
    def _detect_otsu(self, frame):
        """Global Otsu threshold — good when card is main foreground object."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, otsu = cv2.threshold(blurred, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates = []

        edges = cv2.Canny(otsu, 50, 150)
        candidates += self._find_quads_from_edges(edges, frame)

        inv = cv2.bitwise_not(otsu)
        edges_inv = cv2.Canny(inv, 50, 150)
        candidates += self._find_quads_from_edges(edges_inv, frame)

        return candidates

    # ------------------------------------------------------------------
    # Contour processing
    # ------------------------------------------------------------------
    def _process_contour(self, contour, frame):
        """Check if a contour is a card and extract it."""
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None

        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        # Try multiple epsilon values
        for eps_mult in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, eps_mult * peri, True)
            if len(approx) == 4:
                break
        else:
            return None

        if len(approx) != 4:
            return None

        # Check if roughly convex
        if not cv2.isContourConvex(approx):
            return None

        # Check aspect ratio matches a card
        corners = approx.reshape(4, 2).astype(np.float32)
        if not self._check_aspect_ratio(corners):
            return None

        # Check angles are roughly 90 degrees
        if not self._check_angles(corners):
            return None

        # Order corners
        ordered = self._order_corners(corners)

        # Perspective-correct the card
        cropped = self._perspective_transform(frame, ordered)

        # Compute bounding rect
        x, y, w, h = cv2.boundingRect(approx)

        return {
            'corners': ordered,
            'cropped': cropped,
            'bbox': (x, y, w, h),
            'area': area,
        }

    def _check_aspect_ratio(self, corners):
        """Check if the quadrilateral has card-like proportions."""
        ordered = self._order_corners(corners)

        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bot = np.linalg.norm(ordered[2] - ordered[3])
        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])

        avg_width = (width_top + width_bot) / 2
        avg_height = (height_left + height_right) / 2

        if avg_height == 0 or avg_width == 0:
            return False

        ratio = min(avg_width, avg_height) / max(avg_width, avg_height)

        # Card ratio is ~0.716 — allow generous tolerance
        expected = config.CARD_ASPECT_RATIO
        tol = config.CARD_ASPECT_TOLERANCE + 0.05  # Extra tolerance

        return abs(ratio - expected) < tol

    def _check_angles(self, corners):
        """Check that all angles are roughly 90 degrees (between 60 and 120)."""
        ordered = self._order_corners(corners)

        for i in range(4):
            p1 = ordered[i]
            p2 = ordered[(i + 1) % 4]
            p3 = ordered[(i + 2) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            if angle < 55 or angle > 125:
                return False

        return True

    def _order_corners(self, pts):
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]

        return rect

    def _perspective_transform(self, frame, corners, output_width=300, output_height=420):
        """Apply a perspective transform to extract and flatten the card."""
        width_top = np.linalg.norm(corners[1] - corners[0])
        height_left = np.linalg.norm(corners[3] - corners[0])

        if width_top > height_left:
            output_width, output_height = output_height, output_width

        dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(frame, matrix, (output_width, output_height))

        h, w = warped.shape[:2]
        if w > h:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def _deduplicate(self, candidates, frame):
        """
        Remove duplicate detections that overlap significantly.
        Keep the detection with the largest area.
        """
        if not candidates:
            return []

        # Sort by area (largest first)
        candidates.sort(key=lambda c: c.get('area', 0), reverse=True)

        kept = []
        for candidate in candidates:
            bbox = candidate['bbox']
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2

            is_duplicate = False
            for existing in kept:
                ex_bbox = existing['bbox']
                ex_cx = ex_bbox[0] + ex_bbox[2] / 2
                ex_cy = ex_bbox[1] + ex_bbox[3] / 2

                # Check center distance relative to card size
                dist = np.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                avg_size = (bbox[2] + bbox[3] + ex_bbox[2] + ex_bbox[3]) / 4

                if dist < avg_size * 0.3:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(candidate)

        # Re-crop from original frame for best quality
        for det in kept:
            det['cropped'] = self._perspective_transform(frame, det['corners'])

        return kept

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_detections(self, frame, detections, results=None):
        """Draw bounding boxes and labels on the frame."""
        for i, det in enumerate(detections):
            corners = det['corners'].astype(int)
            x, y, w, h = det['bbox']

            # Draw quadrilateral
            cv2.polylines(frame, [corners], True, config.COLOR_BOX, 2)

            label_y = max(y - 10, 20)
            if results and i < len(results):
                r = results[i]
                name = r.get('name', 'Unknown')
                conf = r.get('confidence', 0.0)
                grade = r.get('grade', '?')
                card_num = r.get('card_number', '')
                grade_color = self._grade_color(grade)

                label = f"{name}"
                if card_num:
                    label += f" [{card_num}]"
                label += f" ({conf:.0%}) | {grade}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               0.55, 2)
                cv2.rectangle(frame, (x, label_y - th - 6),
                              (x + tw + 8, label_y + 4), config.COLOR_TEXT_BG, -1)
                cv2.putText(frame, label, (x + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, grade_color, 2)
            else:
                cv2.putText(frame, "Card Detected", (x + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_BOX, 2)

        return frame

    def _grade_color(self, grade):
        """Return a color for the given grade string."""
        colors = {
            "Mint": config.COLOR_GRADE_MINT,
            "Near Mint": config.COLOR_GRADE_NM,
            "Excellent": config.COLOR_GRADE_EX,
            "Good": config.COLOR_GRADE_GOOD,
            "Poor": config.COLOR_GRADE_POOR,
        }
        return colors.get(grade, config.COLOR_TEXT)
