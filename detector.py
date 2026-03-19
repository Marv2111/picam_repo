"""
Card detection module.

Primary: Guide frame — crops whatever is inside the guide rectangle.
Bonus: Auto-detect — finds the single biggest card-like rectangle.
"""

import cv2
import numpy as np
import config


class CardDetector:
    """Card detection with guide frame and auto-detect."""

    def get_guide_region(self, frame):
        """
        Calculate the guide frame rectangle coordinates.
        Returns (x, y, w, h) in pixel coordinates.
        """
        fh, fw = frame.shape[:2]

        guide_w = int(fw * config.GUIDE_WIDTH_PCT)
        guide_h = int(guide_w / config.GUIDE_ASPECT)

        # Make sure it fits
        if guide_h > fh * 0.85:
            guide_h = int(fh * 0.85)
            guide_w = int(guide_h * config.GUIDE_ASPECT)

        gx = (fw - guide_w) // 2
        gy = (fh - guide_h) // 2

        return gx, gy, guide_w, guide_h

    def crop_guide_region(self, frame):
        """Crop the guide frame region from the frame."""
        gx, gy, gw, gh = self.get_guide_region(frame)
        cropped = frame[gy:gy+gh, gx:gx+gw].copy()
        # Resize to standard card proportions
        return cv2.resize(cropped, (300, 420))

    def auto_detect(self, frame):
        """
        Try to find the single biggest card-like rectangle in the frame.
        Returns a detection dict or None.
        """
        h, w = frame.shape[:2]
        min_area = h * w * config.CARD_MIN_AREA_RATIO
        max_area = h * w * config.CARD_MAX_AREA_RATIO

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        best = None
        best_area = 0

        # Try a few edge detection approaches
        for low, high in [(40, 120), (80, 200)]:
            edges = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area or area <= best_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue

                corners = approx.reshape(4, 2).astype(np.float32)
                if not self._check_card_shape(corners):
                    continue

                best_area = area
                best = {
                    'corners': self._order_corners(corners),
                    'bbox': cv2.boundingRect(approx),
                    'area': area,
                }

        if best:
            best['cropped'] = self._perspective_transform(frame, best['corners'])

        return best

    def _check_card_shape(self, corners):
        """Check aspect ratio and angles."""
        ordered = self._order_corners(corners)

        # Aspect ratio check
        w_top = np.linalg.norm(ordered[1] - ordered[0])
        w_bot = np.linalg.norm(ordered[2] - ordered[3])
        h_left = np.linalg.norm(ordered[3] - ordered[0])
        h_right = np.linalg.norm(ordered[2] - ordered[1])

        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2
        if avg_h == 0 or avg_w == 0:
            return False

        ratio = min(avg_w, avg_h) / max(avg_w, avg_h)
        if abs(ratio - config.CARD_ASPECT_RATIO) > config.CARD_ASPECT_TOLERANCE:
            return False

        # Angle check — all corners near 90°
        for i in range(4):
            p1 = ordered[i]
            p2 = ordered[(i + 1) % 4]
            p3 = ordered[(i + 2) % 4]
            v1, v2 = p1 - p2, p3 - p2
            dot = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norms < 1e-6:
                return False
            angle = np.degrees(np.arccos(np.clip(dot / norms, -1, 1)))
            if angle < 65 or angle > 115:
                return False

        return True

    def _order_corners(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _perspective_transform(self, frame, corners):
        ow, oh = 300, 420
        w_top = np.linalg.norm(corners[1] - corners[0])
        h_left = np.linalg.norm(corners[3] - corners[0])
        if w_top > h_left:
            ow, oh = oh, ow

        dst = np.array([[0, 0], [ow-1, 0], [ow-1, oh-1], [0, oh-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(frame, M, (ow, oh))

        h, w = warped.shape[:2]
        if w > h:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped

    def draw_guide(self, frame, active=False):
        """Draw the guide frame overlay on the frame."""
        gx, gy, gw, gh = self.get_guide_region(frame)
        color = config.COLOR_GUIDE_ACTIVE if active else config.COLOR_GUIDE
        thickness = 3 if active else 2

        # Draw corner brackets instead of a full rectangle (looks cleaner)
        bracket = min(gw, gh) // 6

        # Top-left
        cv2.line(frame, (gx, gy), (gx + bracket, gy), color, thickness)
        cv2.line(frame, (gx, gy), (gx, gy + bracket), color, thickness)
        # Top-right
        cv2.line(frame, (gx + gw, gy), (gx + gw - bracket, gy), color, thickness)
        cv2.line(frame, (gx + gw, gy), (gx + gw, gy + bracket), color, thickness)
        # Bottom-left
        cv2.line(frame, (gx, gy + gh), (gx + bracket, gy + gh), color, thickness)
        cv2.line(frame, (gx, gy + gh), (gx, gy + gh - bracket), color, thickness)
        # Bottom-right
        cv2.line(frame, (gx + gw, gy + gh), (gx + gw - bracket, gy + gh), color, thickness)
        cv2.line(frame, (gx + gw, gy + gh), (gx + gw, gy + gh - bracket), color, thickness)

        if not active:
            cv2.putText(frame, "Place card here", (gx + gw // 2 - 80, gy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def draw_auto_detection(self, frame, detection, result=None):
        """Draw auto-detected card outline."""
        if detection is None:
            return frame

        corners = detection['corners'].astype(int)
        cv2.polylines(frame, [corners], True, config.COLOR_BOX, 2)

        if result:
            x, y, w, h = detection['bbox']
            name = result.get('name', '?')
            label = f"[Auto] {name}"
            cv2.putText(frame, label, (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_BOX, 2)

        return frame
