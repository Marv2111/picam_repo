"""
Card identification via OCR.
Reads the Pokémon name (top) and card number (bottom) from a cropped card image.
Supports English and German names (Gen 1-6).
"""

import re
import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("[Identifier] WARNING: pytesseract not installed!")
    print("  sudo apt install tesseract-ocr && pip install pytesseract")

from pokemon_names import POKEMON_ALL, ALL_NAMES, NAME_TO_ENGLISH, ENGLISH_NAMES

CARD_SUFFIXES = ["EX", "GX", "V", "VMAX", "VSTAR", "ex", "BREAK"]


class CardIdentifier:
    """Identify Pokémon cards via OCR."""

    def __init__(self):
        self.lookup = NAME_TO_ENGLISH
        print(f"[Identifier] {len(POKEMON_ALL)} Pokémon loaded (EN+DE), "
              f"OCR {'available' if TESSERACT_AVAILABLE else 'NOT available'}")

    def identify(self, card_image):
        """
        Read name and card number from a cropped card image.
        Tries both orientations in case card is upside down.
        """
        card = cv2.resize(card_image, (300, 420))

        # Try normal orientation
        name_result = self._read_name(card)
        card_number = self._read_card_number(card)

        # If name failed, try flipped (card might be upside down)
        if name_result["confidence"] < 0.5:
            flipped = cv2.rotate(card, cv2.ROTATE_180)
            flip_name = self._read_name(flipped)
            flip_number = self._read_card_number(flipped)
            print(f"[OCR] Trying flipped: name='{flip_name['raw_ocr']}' num='{flip_number}'")
            if flip_name["confidence"] > name_result["confidence"]:
                name_result = flip_name
                card_number = flip_number

        # Save debug image on each scan
        cv2.imwrite('static/debug_card.jpg', card)
        print(f"[Scan] Final: name='{name_result['name']}' number='{card_number}'")

        return {
            "name": name_result["name"],
            "confidence": name_result["confidence"],
            "card_number": card_number,
            "raw_ocr": name_result["raw_ocr"],
        }

    # ------------------------------------------------------------------
    # Name reading
    # ------------------------------------------------------------------
    def _read_name(self, card):
        """Read the Pokémon name from the top of the card."""
        if not TESSERACT_AVAILABLE:
            return {"name": "No OCR", "confidence": 0.0, "raw_ocr": ""}

        h, w = card.shape[:2]
        region = card[int(h * 0.03):int(h * 0.22), int(w * 0.02):int(w * 0.85)]

        cv2.imwrite('static/debug_name.jpg', region)
        variants = self._preprocess(region)

        best_name = "Unknown"
        best_conf = 0.0
        best_raw = ""
        all_reads = []

        for img in variants:
            for psm in ["--psm 7", "--psm 6", "--psm 8", "--psm 13"]:
                try:
                    text = pytesseract.image_to_string(img, config=psm).strip()
                    if not text:
                        continue

                    text_alpha = re.sub(r'[^a-zA-Z.\' -]', '', text).strip()
                    all_reads.append(f"{psm[-2:]}: '{text}' -> '{text_alpha}'")

                    if len(text_alpha) < 3:
                        continue

                    name, conf = self._fuzzy_match(text_alpha)
                    if name:
                        print(f"[OCR name] matched: {name} ({conf:.0%})")
                    if name and conf > best_conf:
                        best_name = name
                        best_conf = conf
                        best_raw = text_alpha
                except Exception:
                    continue

        print(f"[OCR name] All reads: {all_reads}")
        print(f"[OCR name] Best: '{best_name}' ({best_conf:.0%}) raw='{best_raw}'")

        return {
            "name": best_name,
            "confidence": best_conf,
            "raw_ocr": best_raw if best_raw else " | ".join(all_reads[:4]),
        }

    # ------------------------------------------------------------------
    # Card number reading
    # ------------------------------------------------------------------
    def _read_card_number(self, card):
        """Read the card number (e.g. '055/217') from the bottom of the card."""
        if not TESSERACT_AVAILABLE:
            return ""

        h, w = card.shape[:2]

        strips = [
            card[int(h * 0.88):int(h * 0.96), int(w * 0.01):int(w * 0.55)],
            card[int(h * 0.85):int(h * 0.99), int(w * 0.01):int(w * 0.55)],
            card[int(h * 0.90):int(h * 0.99), int(w * 0.01):int(w * 0.55)],
        ]

        # Save debug of first strip
        cv2.imwrite('static/debug_number.jpg', strips[0])

        for region in strips:
            rh, rw = region.shape[:2]
            if rh < 3 or rw < 10:
                continue

            scale = max(6, 250 // max(rh, 1))
            large = cv2.resize(region, (rw * scale, rh * scale),
                               interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

            images = []
            _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            images += [t1, cv2.bitwise_not(t1)]
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            _, t2 = cv2.threshold(clahe.apply(gray), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            images += [t2, cv2.bitwise_not(t2)]

            # No whitelist — read everything, then search for pattern
            for img in images:
                for psm in ["--psm 6", "--psm 7", "--psm 4", "--psm 3"]:
                    try:
                        text = pytesseract.image_to_string(img, config=psm)
                        text = text.replace('\\', '/').replace('|', '/').replace('l', '1').replace('O', '0').replace('o', '0')
                        if text.strip():
                            print(f"[OCR num] '{text.strip()}'")
                        match = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', text)
                        if match:
                            result = f"{match.group(1)}/{match.group(2)}"
                            print(f"[OCR num] FOUND: {result}")
                            return result
                    except Exception:
                        continue

        print("[OCR num] No number found")
        return ""

    # ------------------------------------------------------------------
    # Image preprocessing for OCR
    # ------------------------------------------------------------------
    def _preprocess(self, region):
        h, w = region.shape[:2]
        if h < 5 or w < 10:
            return []

        scale = max(3, 100 // max(h, 1))
        large = cv2.resize(region, (w * scale, h * scale),
                           interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

        results = []
        _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(t1)
        results.append(cv2.bitwise_not(t1))

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, t2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(t2)

        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
        results.append(adapt)

        return results

    # ------------------------------------------------------------------
    # Fuzzy matching
    # ------------------------------------------------------------------
    def _fuzzy_match(self, ocr_text):
        """Match OCR text against known Pokémon names (EN + DE)."""
        text = ocr_text.strip()
        for suffix in CARD_SUFFIXES:
            text = text.replace(suffix, "").strip()
        text = re.sub(r'[^a-zA-Z.\' -]', '', text).strip()

        if len(text) < 3:
            return None, 0.0

        text_lower = text.lower()
        text_clean = re.sub(r'[^a-z]', '', text_lower)

        # Exact match
        if text_lower in self.lookup:
            return self.lookup[text_lower], 0.95
        if text_clean in self.lookup:
            return self.lookup[text_clean], 0.93

        # Substring: check if any Pokémon name is inside the OCR text
        for name in ALL_NAMES:
            name_lower = name.lower()
            name_clean = re.sub(r'[^a-z]', '', name_lower)
            if len(name_clean) >= 4 and name_clean in text_clean:
                en_name = self.lookup.get(name_lower, name)
                print(f"[Match] Found '{name}' inside '{text}'")
                return en_name, 0.90

        # Reverse: check if OCR text is inside a Pokémon name
        if len(text_clean) >= 4:
            for name in ALL_NAMES:
                name_clean = re.sub(r'[^a-z]', '', name.lower())
                if text_clean in name_clean:
                    en_name = self.lookup.get(name.lower(), name)
                    return en_name, 0.80

        # Word-by-word: try each word separately
        words = text.split()
        for word in words:
            word_clean = re.sub(r'[^a-z]', '', word.lower())
            if len(word_clean) >= 4 and word_clean in self.lookup:
                return self.lookup[word_clean], 0.88

        # Edit distance as last resort
        best_name, best_dist = None, 999
        for word in words:
            word_clean = re.sub(r'[^a-z]', '', word.lower())
            if len(word_clean) < 3:
                continue
            for key, name in self.lookup.items():
                if abs(len(key) - len(word_clean)) > 2:
                    continue
                d = self._edit_dist(word_clean, key)
                if d < best_dist:
                    best_dist = d
                    best_name = name

        if best_dist <= 1 and best_name:
            return best_name, 0.80
        if best_dist <= 2 and best_name:
            return best_name, 0.65
        return None, 0.0

    def _edit_dist(self, s1, s2):
        if len(s1) < len(s2):
            return self._edit_dist(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
            prev = curr
        return prev[-1]

    def get_card_count(self):
        return len(POKEMON_ALL)
