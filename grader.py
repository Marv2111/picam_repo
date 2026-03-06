"""
Card condition grading engine.
Analyzes a cropped card image for physical defects and produces a grade.

Defect categories:
  - Edge whitening: worn edges showing white cardboard
  - Scratches: thin surface marks
  - Dirt/stains: dark spots or discoloration
  - Bends/creases: fold lines
  - Centering: uneven borders around the card artwork
"""

import cv2
import numpy as np
import config


class CardGrader:
    """Analyze card condition and produce a grade."""

    def grade(self, card_image):
        """
        Grade a cropped, perspective-corrected card image.

        Args:
            card_image: BGR image (ideally 300x420)

        Returns:
            dict with:
                - 'grade': str ('Mint', 'Near Mint', etc.)
                - 'score': float 0.0 (perfect) to 1.0 (destroyed)
                - 'defects': dict of individual defect scores
                - 'details': dict of human-readable defect descriptions
        """
        # Ensure consistent size
        card = cv2.resize(card_image, (300, 420))
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

        # Analyze each defect category
        edge_score, edge_detail = self._analyze_edge_whitening(card, gray)
        scratch_score, scratch_detail = self._analyze_scratches(gray)
        dirt_score, dirt_detail = self._analyze_dirt(card, gray)
        bend_score, bend_detail = self._analyze_bends(gray)
        center_score, center_detail = self._analyze_centering(gray)

        # Weighted combination
        overall = (
            config.WEIGHT_EDGE_WHITENING * edge_score +
            config.WEIGHT_SCRATCHES * scratch_score +
            config.WEIGHT_DIRT * dirt_score +
            config.WEIGHT_BENDS * bend_score +
            config.WEIGHT_CENTERING * center_score
        )
        overall = np.clip(overall, 0.0, 1.0)

        # Determine grade from overall score
        grade_label = "Poor"
        for label, (low, high) in config.GRADE_THRESHOLDS.items():
            if low <= overall < high:
                grade_label = label
                break

        defects = {
            'edge_whitening': round(edge_score, 3),
            'scratches': round(scratch_score, 3),
            'dirt': round(dirt_score, 3),
            'bends': round(bend_score, 3),
            'centering': round(center_score, 3),
        }

        details = {
            'edge_whitening': edge_detail,
            'scratches': scratch_detail,
            'dirt': dirt_detail,
            'bends': bend_detail,
            'centering': center_detail,
        }

        return {
            'grade': grade_label,
            'score': round(overall, 3),
            'defects': defects,
            'details': details,
        }

    # -----------------------------------------------------------------
    # Edge Whitening
    # -----------------------------------------------------------------
    def _analyze_edge_whitening(self, card, gray):
        """
        Check for white/light pixels along the card borders.
        Worn edges expose white cardboard underneath the colored surface.
        """
        h, w = gray.shape
        border = int(min(h, w) * config.EDGE_BORDER_WIDTH_FRAC)
        border = max(border, 3)

        # Extract border strips
        top = gray[0:border, :]
        bottom = gray[h - border:h, :]
        left = gray[:, 0:border]
        right = gray[:, w - border:w]

        # Count pixels above the whiteness threshold
        white_pixels = 0
        total_pixels = 0
        for strip in [top, bottom, left, right]:
            white_pixels += np.count_nonzero(strip > config.EDGE_WHITE_THRESHOLD)
            total_pixels += strip.size

        if total_pixels == 0:
            return 0.0, "No border data"

        white_ratio = white_pixels / total_pixels
        score = np.clip(white_ratio * config.EDGE_WHITE_SCORE_SCALE, 0.0, 1.0)

        if score < 0.1:
            detail = "Edges clean"
        elif score < 0.3:
            detail = "Minor edge whitening"
        elif score < 0.6:
            detail = "Moderate edge whitening"
        else:
            detail = "Severe edge whitening"

        return score, detail

    # -----------------------------------------------------------------
    # Scratches
    # -----------------------------------------------------------------
    def _analyze_scratches(self, gray):
        """
        Detect surface scratches using a high-pass filter.
        Scratches appear as thin bright lines in the high-frequency response.
        """
        # Apply a high-pass filter (original minus blurred)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        high_pass = cv2.absdiff(gray, blurred)

        # Threshold to find strong high-frequency responses
        _, thresh = cv2.threshold(high_pass, config.SCRATCH_THRESHOLD,
                                  255, cv2.THRESH_BINARY)

        # Morphological operations to connect scratch segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (config.SCRATCH_KERNEL_SIZE,
                                            config.SCRATCH_KERNEL_SIZE))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Score based on fraction of pixels detected
        scratch_ratio = np.count_nonzero(cleaned) / cleaned.size
        score = np.clip(scratch_ratio * config.SCRATCH_SCORE_SCALE, 0.0, 1.0)

        if score < 0.1:
            detail = "No scratches detected"
        elif score < 0.3:
            detail = "Minor surface marks"
        elif score < 0.6:
            detail = "Moderate scratching"
        else:
            detail = "Heavy scratching"

        return score, detail

    # -----------------------------------------------------------------
    # Dirt / Stains
    # -----------------------------------------------------------------
    def _analyze_dirt(self, card, gray):
        """
        Detect dirt or stains by looking for dark spots on lighter card regions.
        Uses local contrast analysis to find anomalous dark patches.
        """
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)

        # Focus on the inner area (exclude borders which have their own analysis)
        h, w = gray.shape
        margin = int(min(h, w) * 0.08)
        inner_gray = gray[margin:h - margin, margin:w - margin]
        inner_val = val[margin:h - margin, margin:w - margin]

        # Local mean for context
        local_mean = cv2.blur(inner_gray, (31, 31))

        # Dirt = significantly darker than local context
        diff = local_mean.astype(np.float32) - inner_gray.astype(np.float32)
        dirt_mask = diff > config.DIRT_DARK_THRESHOLD

        dirt_ratio = np.count_nonzero(dirt_mask) / dirt_mask.size
        score = np.clip(dirt_ratio * config.DIRT_SCORE_SCALE, 0.0, 1.0)

        if score < 0.1:
            detail = "Surface clean"
        elif score < 0.3:
            detail = "Minor spots detected"
        elif score < 0.6:
            detail = "Noticeable dirt or stains"
        else:
            detail = "Significant contamination"

        return score, detail

    # -----------------------------------------------------------------
    # Bends / Creases
    # -----------------------------------------------------------------
    def _analyze_bends(self, gray):
        """
        Detect bends and creases using line detection.
        Creases appear as long, straight edges that span across the card.
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Remove edges near the card border (those are the card edges)
        h, w = edges.shape
        margin = int(min(h, w) * 0.06)
        mask = np.zeros_like(edges)
        mask[margin:h - margin, margin:w - margin] = 255
        edges = cv2.bitwise_and(edges, mask)

        # Detect straight lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=config.BEND_LINE_THRESHOLD,
            minLineLength=config.BEND_MIN_LINE_LENGTH,
            maxLineGap=10
        )

        if lines is None:
            return 0.0, "No bends detected"

        # Filter for long lines (potential creases)
        long_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > min(h, w) * 0.2:
                long_lines += 1

        # Score based on number of long internal lines
        score = np.clip(long_lines / 5.0 * config.BEND_SCORE_SCALE, 0.0, 1.0)

        if long_lines == 0:
            detail = "No bends detected"
        elif long_lines <= 2:
            detail = f"Possible bend ({long_lines} line(s))"
        else:
            detail = f"Multiple creases ({long_lines} lines)"

        return score, detail

    # -----------------------------------------------------------------
    # Centering
    # -----------------------------------------------------------------
    def _analyze_centering(self, gray):
        """
        Check if the card's artwork is centered within its borders.
        Measures the border widths on all four sides and checks symmetry.
        """
        h, w = gray.shape

        # Use thresholding to find the border vs artwork region
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the bounding box of the main content (artwork area)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0, "Could not analyze centering"

        # Get the largest internal contour (the artwork boundary)
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)

        # Compute border widths
        border_left = x
        border_right = w - (x + cw)
        border_top = y
        border_bottom = h - (y + ch)

        # Check horizontal and vertical centering
        h_total = border_left + border_right
        v_total = border_top + border_bottom

        if h_total == 0 or v_total == 0:
            return 0.0, "Could not measure borders"

        h_ratio = abs(border_left - border_right) / max(h_total, 1)
        v_ratio = abs(border_top - border_bottom) / max(v_total, 1)

        avg_deviation = (h_ratio + v_ratio) / 2
        score = np.clip(avg_deviation / config.CENTER_TOLERANCE, 0.0, 1.0)

        lr_pct = f"{border_left / max(h_total, 1) * 100:.0f}/{border_right / max(h_total, 1) * 100:.0f}"
        tb_pct = f"{border_top / max(v_total, 1) * 100:.0f}/{border_bottom / max(v_total, 1) * 100:.0f}"

        if score < 0.2:
            detail = f"Well centered (LR: {lr_pct}, TB: {tb_pct})"
        elif score < 0.5:
            detail = f"Slightly off-center (LR: {lr_pct}, TB: {tb_pct})"
        else:
            detail = f"Notably off-center (LR: {lr_pct}, TB: {tb_pct})"

        return score, detail
