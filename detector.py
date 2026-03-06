"""
Card detection module.
Finds rectangular card-shaped objects in a camera frame using contour analysis.
Returns bounding quadrilaterals and perspective-corrected (flattened) card images.
"""

import cv2
import numpy as np
import config


class CardDetector:
    """Detect rectangular cards in a frame using contour-based analysis."""

    def __init__(self):
        self.min_area = None  # Set dynamically based on frame size
        self.max_area = None

    def detect(self, frame):
        """
        Find cards in the given BGR frame.

        Returns:
            list of dict, each containing:
                - 'corners': 4 corner points of the card in the original frame
                - 'cropped': perspective-corrected card image (BGR)
                - 'bbox': (x, y, w, h) bounding rectangle
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        self.min_area = frame_area * config.CARD_MIN_AREA_RATIO
        self.max_area = frame_area * config.CARD_MAX_AREA_RATIO

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)

        # Edge detection
        edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)

        # Dilate to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        cards = []
        for contour in contours:
            card = self._process_contour(contour, frame)
            if card is not None:
                cards.append(card)

        return cards

    def _process_contour(self, contour, frame):
        """Check if a contour is a card and extract it."""
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None

        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, config.CONTOUR_APPROX_EPSILON * peri, True)

        # Must be a quadrilateral
        if len(approx) != 4:
            return None

        # Check if roughly convex
        if not cv2.isContourConvex(approx):
            return None

        # Check aspect ratio matches a card
        corners = approx.reshape(4, 2).astype(np.float32)
        if not self._check_aspect_ratio(corners):
            return None

        # Order corners: top-left, top-right, bottom-right, bottom-left
        ordered = self._order_corners(corners)

        # Perspective-correct the card
        cropped = self._perspective_transform(frame, ordered)

        # Compute simple bounding rect for display
        x, y, w, h = cv2.boundingRect(approx)

        return {
            'corners': ordered,
            'cropped': cropped,
            'bbox': (x, y, w, h),
        }

    def _check_aspect_ratio(self, corners):
        """Check if the quadrilateral has card-like proportions."""
        ordered = self._order_corners(corners)

        # Compute side lengths
        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bot = np.linalg.norm(ordered[2] - ordered[3])
        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])

        avg_width = (width_top + width_bot) / 2
        avg_height = (height_left + height_right) / 2

        if avg_height == 0:
            return False

        ratio = avg_width / avg_height

        # Card could be portrait or landscape
        expected = config.CARD_ASPECT_RATIO
        tol = config.CARD_ASPECT_TOLERANCE

        portrait_ok = abs(ratio - expected) < tol
        landscape_ok = abs(ratio - (1.0 / expected)) < tol

        return portrait_ok or landscape_ok

    def _order_corners(self, pts):
        """
        Order 4 points as: top-left, top-right, bottom-right, bottom-left.
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # Top-left has smallest x+y
        rect[2] = pts[np.argmax(s)]   # Bottom-right has largest x+y

        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]   # Top-right has smallest x-y
        rect[3] = pts[np.argmax(d)]   # Bottom-left has largest x-y

        return rect

    def _perspective_transform(self, frame, corners, output_width=300, output_height=420):
        """
        Apply a perspective transform to extract and flatten the card.
        Output size matches standard card proportions.
        """
        # Determine if landscape — if so, swap target dimensions
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

        # Ensure portrait orientation (height > width)
        h, w = warped.shape[:2]
        if w > h:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    def draw_detections(self, frame, detections, results=None):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: BGR image to draw on (modified in place)
            detections: list from detect()
            results: optional list of dicts with 'name', 'confidence', 'grade', 'defects'
        """
        for i, det in enumerate(detections):
            corners = det['corners'].astype(int)
            x, y, w, h = det['bbox']

            # Draw quadrilateral
            cv2.polylines(frame, [corners], True, config.COLOR_BOX, 2)

            # Draw label background
            label_y = max(y - 10, 20)
            if results and i < len(results):
                r = results[i]
                name = r.get('name', 'Unknown')
                conf = r.get('confidence', 0.0)
                grade = r.get('grade', '?')
                grade_color = self._grade_color(grade)

                label = f"{name} ({conf:.0%}) | {grade}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               0.6, 2)
                cv2.rectangle(frame, (x, label_y - th - 6),
                              (x + tw + 8, label_y + 4), config.COLOR_TEXT_BG, -1)
                cv2.putText(frame, label, (x + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, grade_color, 2)
            else:
                label = "Card Detected"
                cv2.putText(frame, label, (x + 4, label_y),
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
