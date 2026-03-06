"""
Card identification module.
Uses ORB feature matching to identify which Pokémon card is detected
by comparing against a pre-built reference database.
"""

import os
import pickle
import cv2
import numpy as np
import config


class CardIdentifier:
    """Identify Pokémon cards using ORB feature matching."""

    def __init__(self, db_path=None, ref_dir=None):
        self.db_path = db_path or config.REFERENCE_DB_PATH
        self.ref_dir = ref_dir or config.REFERENCE_CARDS_DIR
        self.orb = cv2.ORB_create(nfeatures=config.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.database = {}  # name -> {'descriptors': ..., 'keypoints_count': ...}
        self._load_database()

    def _load_database(self):
        """Load pre-computed feature database, or build it if missing."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.database = pickle.load(f)
            print(f"[Identifier] Loaded {len(self.database)} reference cards "
                  f"from {self.db_path}")
        elif os.path.isdir(self.ref_dir):
            print("[Identifier] No database found. Building from reference images...")
            self.build_database()
        else:
            print(f"[Identifier] WARNING: No reference cards found. "
                  f"Place card images in '{self.ref_dir}/' and run "
                  f"'python tools/build_reference_db.py'")

    def build_database(self):
        """
        Scan the reference_cards directory, compute ORB features for each image,
        and save the database.
        """
        if not os.path.isdir(self.ref_dir):
            print(f"[Identifier] Creating directory: {self.ref_dir}/")
            os.makedirs(self.ref_dir, exist_ok=True)
            return

        self.database = {}
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        for filename in sorted(os.listdir(self.ref_dir)):
            if not filename.lower().endswith(extensions):
                continue

            filepath = os.path.join(self.ref_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"  [SKIP] Could not read: {filename}")
                continue

            # Resize to standard size for consistent matching
            img = cv2.resize(img, (300, 420))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) < 5:
                print(f"  [SKIP] Too few features: {filename}")
                continue

            # Card name derived from filename (without extension)
            name = os.path.splitext(filename)[0]
            # Clean up the name for display
            display_name = name.replace('_', ' ').replace('-', ' ').title()

            self.database[name] = {
                'display_name': display_name,
                'descriptors': descriptors,
                'keypoints_count': len(keypoints),
            }
            print(f"  [OK] {display_name}: {len(keypoints)} features")

        # Save database
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"[Identifier] Saved database with {len(self.database)} cards "
              f"to {self.db_path}")

    def identify(self, card_image):
        """
        Identify a cropped card image.

        Args:
            card_image: BGR image of a cropped, perspective-corrected card.

        Returns:
            dict with:
                - 'name': display name of the matched card (or 'Unknown')
                - 'confidence': float 0.0-1.0
                - 'matches': number of good feature matches
        """
        if not self.database:
            return {'name': 'Unknown (no reference DB)', 'confidence': 0.0,
                    'matches': 0}

        # Compute ORB features for the query card
        resized = cv2.resize(card_image, (300, 420))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < 5:
            return {'name': 'Unknown', 'confidence': 0.0, 'matches': 0}

        best_name = 'Unknown'
        best_display_name = 'Unknown'
        best_good_matches = 0
        best_confidence = 0.0

        for name, ref_data in self.database.items():
            ref_desc = ref_data['descriptors']

            # KNN match
            try:
                raw_matches = self.matcher.knnMatch(descriptors, ref_desc, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test
            good_matches = []
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            num_good = len(good_matches)
            if num_good > best_good_matches:
                best_good_matches = num_good
                best_name = name
                best_display_name = ref_data['display_name']

        # Compute confidence as ratio of good matches to threshold
        if best_good_matches >= config.MIN_GOOD_MATCHES:
            # Scale confidence: MIN_GOOD_MATCHES → 0.5, 3x that → 1.0
            confidence = min(1.0, 0.5 + 0.5 * (best_good_matches - config.MIN_GOOD_MATCHES)
                             / (config.MIN_GOOD_MATCHES * 2))
            return {
                'name': best_display_name,
                'confidence': confidence,
                'matches': best_good_matches,
            }
        else:
            return {
                'name': 'Unknown',
                'confidence': best_good_matches / config.MIN_GOOD_MATCHES * 0.5,
                'matches': best_good_matches,
            }

    def get_card_count(self):
        """Return the number of reference cards in the database."""
        return len(self.database)
