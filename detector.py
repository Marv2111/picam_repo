"""
Card detection module with voting system.
Uses multiple detection strategies but ONLY reports a card if at least
2 strategies agree on the same region. This prevents false positives
while still working on varied backgrounds.
"""

import cv2
import numpy as np
import config


# Minimum number of strategies that must detect the same region
MIN_VOTES = 2


class CardDetector:
    """Detect cards using multi-strategy voting."""

    def __init__(self):
        self.min_area = None
        self.max_area = None

    def detect(self, frame):
        """
        Find cards in the given BGR frame.

        Returns:
            list of dict with 'corners', 'cropped', 'bbox'
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        self.min_area = frame_area * config.CARD_MIN_AREA_RATIO
        self.max_area = frame_area * config.CARD_MAX_AREA_RATIO

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Run each strategy independently — collect all candidate bboxes
        all_candidates = []

        # Strategy 1: Canny (moderate threshold)
        s1 = self._detect_canny(blurred, frame, low=40, high=120)
        for c in s1:
            c['strategy'] = 1
        all_candidates += s1

        # Strategy 2: Canny (high threshold — strong edges only)
        s2 = self._detect_canny(blurred, frame, low=80, high=200)
        for c in s2:
            c['strategy'] = 2
        all_candidates += s2

        # Strategy 3: Adaptive threshold
        s3 = self._detect_adaptive(blurred, frame)
        for c in s3:
            c['strategy'] = 3
        all_candidates += s3

        # Strategy 4: Otsu threshold
        s4 = self._detect_otsu(blurred, frame)
        for c in s4:
            c['strategy'] = 4
        all_candidates += s4

        if not all_candidates:
            return []

        # Vote: group overlapping candidates, keep those with >= MIN_VOTES
        cards = self._vote_and_filter(all_candidates, frame)

        return cards

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------
    def _detect_canny(self, blurred, frame, low, high):
        """Find card quads using Canny edge detection."""
        edges = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return self._find_quads(edges, frame)

    def _detect_adaptive(self, blurred, frame):
        """Find card quads using adaptive thresholding."""
        candidates = []
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 5)
        edges = cv2.Canny(thresh, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        candidates += self._find_quads(edges, frame)
        return candidates

    def _detect_otsu(self, blurred, frame):
        """Find card quads using Otsu thresholding."""
        _, otsu = cv2.threshold(blurred, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(otsu, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return self._find_quads(edges, frame)

    # ------------------------------------------------------------------
    # Quad finding from edge maps
    # ------------------------------------------------------------------
    def _find_quads(self, edges, frame):
        """Find card-like quadrilaterals from an edge image."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            card = self._process_contour(cnt, frame)
            if card is not None:
                results.append(card)
        return results

    def _process_contour(self, contour, frame):
        """Check if a contour looks like a card."""
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None

        peri = cv2.arcLength(contour, True)

        # Try to approximate to 4 vertices
        for eps in [0.02, 0.03, 0.04]:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4:
                break
        else:
            return None

        if len(approx) != 4:
            return None

        if not cv2.isContourConvex(approx):
            return None

        corners = approx.reshape(4, 2).astype(np.float32)

        if not self._check_aspect_ratio(corners):
            return None

        if not self._check_angles(corners):
            return None

        ordered = self._order_corners(corners)
        x, y, w, h = cv2.boundingRect(approx)

        return {
            'corners': ordered,
            'bbox': (x, y, w, h),
            'area': area,
        }

    # ------------------------------------------------------------------
    # Geometry checks
    # ------------------------------------------------------------------
    def _check_aspect_ratio(self, corners):
        """Check card-like proportions."""
        ordered = self._order_corners(corners)

        w_top = np.linalg.norm(ordered[1] - ordered[0])
        w_bot = np.linalg.norm(ordered[2] - ordered[3])
        h_left = np.linalg.norm(ordered[3] - ordered[0])
        h_right = np.linalg.norm(ordered[2] - ordered[1])

        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2

        if avg_h == 0 or avg_w == 0:
            return False

        ratio = min(avg_w, avg_h) / max(avg_w, avg_h)

        expected = config.CARD_ASPECT_RATIO
        tol = config.CARD_ASPECT_TOLERANCE

        return abs(ratio - expected) < tol

    def _check_angles(self, corners):
        """All interior angles must be close to 90° (between 65° and 115°)."""
        ordered = self._order_corners(corners)

        for i in range(4):
            p1 = ordered[i]
            p2 = ordered[(i + 1) % 4]
            p3 = ordered[(i + 2) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            dot = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norms < 1e-6:
                return False

            cos_angle = np.clip(dot / norms, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))

            if angle < 65 or angle > 115:
                return False

        return True

    def _order_corners(self, pts):
        """Order: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    # ------------------------------------------------------------------
    # Voting: require multiple strategies to agree
    # ------------------------------------------------------------------
    def _vote_and_filter(self, candidates, frame):
        """
        Group overlapping candidates and only keep regions
        detected by at least MIN_VOTES different strategies.
        """
        if not candidates:
            return []

        # Cluster candidates by center proximity
        clusters = []
        for cand in candidates:
            bx, by, bw, bh = cand['bbox']
            cx = bx + bw / 2
            cy = by + bh / 2

            placed = False
            for cluster in clusters:
                ref_cx, ref_cy, ref_size = cluster['center']
                dist = np.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2)

                if dist < ref_size * 0.35:
                    cluster['items'].append(cand)
                    cluster['strategies'].add(cand.get('strategy', 0))
                    # Update center as running average
                    n = len(cluster['items'])
                    cluster['center'] = (
                        (ref_cx * (n-1) + cx) / n,
                        (ref_cy * (n-1) + cy) / n,
                        (ref_size * (n-1) + (bw + bh) / 2) / n,
                    )
                    placed = True
                    break

            if not placed:
                clusters.append({
                    'items': [cand],
                    'strategies': {cand.get('strategy', 0)},
                    'center': (cx, cy, (bw + bh) / 2),
                })

        # Filter: keep clusters with enough strategy votes
        results = []
        for cluster in clusters:
            num_strategies = len(cluster['strategies'])
            if num_strategies >= MIN_VOTES:
                # Pick the candidate with the largest area as representative
                best = max(cluster['items'], key=lambda c: c.get('area', 0))

                # Perspective-correct
                cropped = self._perspective_transform(frame, best['corners'])
                best['cropped'] = cropped
                best['votes'] = num_strategies
                results.append(best)

        return results

    def _perspective_transform(self, frame, corners, output_width=300, output_height=420):
        """Apply perspective transform to flatten the card."""
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
    # Drawing
    # ------------------------------------------------------------------
    def draw_detections(self, frame, detections, results=None):
        """Draw bounding boxes and labels on the frame."""
        for i, det in enumerate(detections):
            corners = det['corners'].astype(int)
            x, y, w, h = det['bbox']

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
        colors = {
            "Mint": config.COLOR_GRADE_MINT,
            "Near Mint": config.COLOR_GRADE_NM,
            "Excellent": config.COLOR_GRADE_EX,
            "Good": config.COLOR_GRADE_GOOD,
            "Poor": config.COLOR_GRADE_POOR,
        }
        return colors.get(grade, config.COLOR_TEXT)
