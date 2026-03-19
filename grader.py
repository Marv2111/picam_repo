"""Card condition grading engine using OpenCV analysis."""

import cv2
import numpy as np
import config


class CardGrader:
    """Analyze card condition and produce a grade."""

    def grade(self, card_image):
        card = cv2.resize(card_image, (300, 420))
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

        e_score, e_detail = self._edge_whitening(card, gray)
        s_score, s_detail = self._scratches(gray)
        d_score, d_detail = self._dirt(card, gray)
        b_score, b_detail = self._bends(gray)
        c_score, c_detail = self._centering(gray)

        overall = (
            config.WEIGHT_EDGE_WHITENING * e_score +
            config.WEIGHT_SCRATCHES * s_score +
            config.WEIGHT_DIRT * d_score +
            config.WEIGHT_BENDS * b_score +
            config.WEIGHT_CENTERING * c_score
        )
        overall = np.clip(overall, 0.0, 1.0)

        grade_label = "Poor"
        for label, (lo, hi) in config.GRADE_THRESHOLDS.items():
            if lo <= overall < hi:
                grade_label = label
                break

        return {
            'grade': grade_label,
            'score': round(float(overall), 3),
            'defects': {
                'edge_whitening': round(float(e_score), 3),
                'scratches': round(float(s_score), 3),
                'dirt': round(float(d_score), 3),
                'bends': round(float(b_score), 3),
                'centering': round(float(c_score), 3),
            },
            'details': {
                'edge_whitening': e_detail,
                'scratches': s_detail,
                'dirt': d_detail,
                'bends': b_detail,
                'centering': c_detail,
            },
        }

    def _edge_whitening(self, card, gray):
        h, w = gray.shape
        border = max(3, int(min(h, w) * 0.04))
        strips = [gray[:border, :], gray[h-border:, :],
                  gray[:, :border], gray[:, w-border:]]
        white = sum(np.count_nonzero(s > 200) for s in strips)
        total = sum(s.size for s in strips)
        ratio = white / max(total, 1)
        score = np.clip(ratio * 3.0, 0.0, 1.0)
        if score < 0.1: d = "Edges clean"
        elif score < 0.3: d = "Minor edge whitening"
        elif score < 0.6: d = "Moderate edge whitening"
        else: d = "Severe edge whitening"
        return score, d

    def _scratches(self, gray):
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        hp = cv2.absdiff(gray, blurred)
        _, thresh = cv2.threshold(hp, 40, 255, cv2.THRESH_BINARY)
        ratio = np.count_nonzero(thresh) / thresh.size
        score = np.clip(ratio * 5.0, 0.0, 1.0)
        if score < 0.1: d = "No scratches"
        elif score < 0.3: d = "Minor surface marks"
        elif score < 0.6: d = "Moderate scratching"
        else: d = "Heavy scratching"
        return score, d

    def _dirt(self, card, gray):
        h, w = gray.shape
        m = int(min(h, w) * 0.08)
        inner = gray[m:h-m, m:w-m]
        local_mean = cv2.blur(inner, (31, 31)).astype(np.float32)
        diff = local_mean - inner.astype(np.float32)
        ratio = np.count_nonzero(diff > 50) / diff.size
        score = np.clip(ratio * 4.0, 0.0, 1.0)
        if score < 0.1: d = "Surface clean"
        elif score < 0.3: d = "Minor spots"
        elif score < 0.6: d = "Noticeable dirt"
        else: d = "Significant contamination"
        return score, d

    def _bends(self, gray):
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        m = int(min(h, w) * 0.06)
        mask = np.zeros_like(edges)
        mask[m:h-m, m:w-m] = 255
        edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)
        if lines is None:
            return 0.0, "No bends detected"
        long = sum(1 for l in lines if np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) > min(h,w)*0.2)
        score = np.clip(long / 5.0 * 4.0, 0.0, 1.0)
        if long == 0: d = "No bends detected"
        elif long <= 2: d = f"Possible bend ({long} line(s))"
        else: d = f"Multiple creases ({long} lines)"
        return score, d

    def _centering(self, gray):
        h, w = gray.shape
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, "Could not analyze"
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        bl, br = x, w - (x + cw)
        bt, bb = y, h - (y + ch)
        ht, vt = bl + br, bt + bb
        if ht == 0 or vt == 0:
            return 0.0, "Could not measure"
        h_dev = abs(bl - br) / max(ht, 1)
        v_dev = abs(bt - bb) / max(vt, 1)
        score = np.clip((h_dev + v_dev) / 2 / 0.15, 0.0, 1.0)
        lr = f"{bl/max(ht,1)*100:.0f}/{br/max(ht,1)*100:.0f}"
        tb = f"{bt/max(vt,1)*100:.0f}/{bb/max(vt,1)*100:.0f}"
        if score < 0.2: d = f"Well centered (LR:{lr} TB:{tb})"
        elif score < 0.5: d = f"Slightly off-center (LR:{lr} TB:{tb})"
        else: d = f"Off-center (LR:{lr} TB:{tb})"
        return score, d
