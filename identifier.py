"""
Pokémon identification module.
Identifies which Pokémon is shown on a card by matching the card's artwork
region against downloaded Pokémon sprites.

Uses a combination of:
  - Color histogram comparison (fast, good for distinctive Pokémon colors)
  - ORB feature matching (catches shape/detail similarities)
"""

import os
import json
import pickle
import cv2
import numpy as np
import config


class CardIdentifier:
    """Identify which Pokémon is on a card using artwork matching."""

    def __init__(self, db_path=None, ref_dir=None):
        self.db_path = db_path or config.REFERENCE_DB_PATH
        self.ref_dir = ref_dir or config.REFERENCE_CARDS_DIR
        self.orb = cv2.ORB_create(nfeatures=config.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.database = {}
        self.name_map = {}
        self._load_name_map()
        self._load_database()

    def _load_name_map(self):
        """Load the Pokémon name mapping file if available."""
        map_path = os.path.join(self.ref_dir, "_name_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                self.name_map = json.load(f)

    def _load_database(self):
        """Load pre-computed feature database, or build if missing."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.database = pickle.load(f)
            print(f"[Identifier] Loaded {len(self.database)} Pokémon from {self.db_path}")
        elif os.path.isdir(self.ref_dir):
            print("[Identifier] No database found. Building from sprites...")
            self.build_database()
        else:
            print(f"[Identifier] WARNING: No sprites found. Run:")
            print(f"  python3 tools/download_pokemon_sprites.py")

    def build_database(self):
        """
        Build a reference database from Pokémon sprite images.
        For each sprite, store:
          - Color histogram (HSV, for color-based matching)
          - ORB descriptors (for shape-based matching)
          - Display name
        """
        if not os.path.isdir(self.ref_dir):
            os.makedirs(self.ref_dir, exist_ok=True)
            return

        self.database = {}
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for filename in sorted(os.listdir(self.ref_dir)):
            if not filename.lower().endswith(extensions):
                continue
            if filename.startswith("_"):
                continue

            filepath = os.path.join(self.ref_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Handle transparency (sprites often have alpha channel)
            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                bgr = img[:, :, :3]
                white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
                img_bgr = (bgr * alpha[:, :, np.newaxis] +
                           white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            else:
                img_bgr = img[:, :, :3] if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize to standard size
            img_resized = cv2.resize(img_bgr, (200, 200))

            # Compute color histogram (HSV)
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            cv2.normalize(hist_v, hist_v)
            color_hist = np.concatenate([hist_h.flatten(),
                                          hist_s.flatten(),
                                          hist_v.flatten()])

            # Compute ORB features
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            # Determine display name
            if filename in self.name_map:
                display_name = self.name_map[filename]
            else:
                name = os.path.splitext(filename)[0]
                display_name = name.replace("_", " ").replace("-", " ").title()
                parts = display_name.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    display_name = parts[0]

            self.database[filename] = {
                "display_name": display_name,
                "color_hist": color_hist,
                "descriptors": descriptors,
                "kp_count": len(keypoints) if keypoints else 0,
            }
            print(f"  [OK] {display_name}")

        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)
        print(f"[Identifier] Saved {len(self.database)} Pokémon to {self.db_path}")

    def _extract_artwork_region(self, card_image):
        """
        Extract the artwork region from a Pokémon card.
        The artwork is roughly in the upper-center portion of the card.
        """
        h, w = card_image.shape[:2]
        y1 = int(h * 0.12)
        y2 = int(h * 0.62)
        x1 = int(w * 0.08)
        x2 = int(w * 0.92)
        return card_image[y1:y2, x1:x2]

    def identify(self, card_image):
        """
        Identify which Pokémon is on the card.

        Args:
            card_image: BGR image of a cropped, perspective-corrected card.

        Returns:
            dict with 'name', 'confidence', 'matches'
        """
        if not self.database:
            return {"name": "No Pokémon DB", "confidence": 0.0, "matches": 0}

        # Extract the artwork area of the card
        artwork = self._extract_artwork_region(card_image)
        artwork_resized = cv2.resize(artwork, (200, 200))

        # Compute color histogram for the artwork
        hsv = cv2.cvtColor(artwork_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)
        query_hist = np.concatenate([hist_h.flatten(),
                                      hist_s.flatten(),
                                      hist_v.flatten()])

        # Compute ORB features for the artwork
        gray = cv2.cvtColor(artwork_resized, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        best_name = "Unknown"
        best_score = -1
        best_orb_matches = 0

        for filename, ref in self.database.items():
            # Color histogram score (correlation: 1.0 = perfect)
            ref_hist = ref["color_hist"]
            color_score = cv2.compareHist(
                query_hist.astype(np.float32).reshape(-1, 1),
                ref_hist.astype(np.float32).reshape(-1, 1),
                cv2.HISTCMP_CORREL
            )

            # ORB feature score
            orb_score = 0.0
            orb_matches = 0
            if desc is not None and ref["descriptors"] is not None:
                try:
                    raw = self.matcher.knnMatch(desc, ref["descriptors"], k=2)
                    good = []
                    for pair in raw:
                        if len(pair) == 2:
                            m, n = pair
                            if m.distance < 0.75 * n.distance:
                                good.append(m)
                    orb_matches = len(good)
                    orb_score = min(orb_matches / 20.0, 1.0)
                except cv2.error:
                    pass

            # Combined: color weighted higher (sprites vs card art differ in detail)
            combined = 0.65 * max(color_score, 0) + 0.35 * orb_score

            if combined > best_score:
                best_score = combined
                best_name = ref["display_name"]
                best_orb_matches = orb_matches

        confidence = float(np.clip(best_score, 0.0, 1.0))

        return {
            "name": best_name,
            "confidence": confidence,
            "matches": best_orb_matches,
        }

    def get_card_count(self):
        """Return the number of Pokémon in the database."""
        return len(self.database)
