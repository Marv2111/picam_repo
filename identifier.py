"""
Pokémon identification module — CNN-based.
Uses MobileNetV2 (via OpenCV DNN) as a feature extractor to create
"fingerprints" of Pokémon artwork, then matches cards against the database
using cosine similarity.

Much more accurate than color histograms because the CNN captures shapes,
patterns, and semantic features — not just colors.

Fallback: If the ONNX model isn't available, uses an improved multi-feature
matching approach (histogram + Hu moments + edge features).
"""

import os
import json
import pickle
import cv2
import numpy as np
import config


class CardIdentifier:
    """Identify which Pokémon is on a card using CNN feature matching."""

    def __init__(self, db_path=None, ref_dir=None, model_path=None):
        self.db_path = db_path or config.REFERENCE_DB_PATH
        self.ref_dir = ref_dir or config.REFERENCE_CARDS_DIR
        self.model_path = model_path or os.path.join("models", "mobilenetv2-12.onnx")
        self.net = None
        self.use_cnn = False
        self.database = {}
        self.name_map = {}

        self._load_model()
        self._load_name_map()
        self._load_database()

    def _load_model(self):
        """Load the MobileNetV2 ONNX model for feature extraction."""
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.use_cnn = True
                print(f"[Identifier] Loaded MobileNetV2 CNN model")
            except Exception as e:
                print(f"[Identifier] Failed to load ONNX model: {e}")
                print(f"[Identifier] Falling back to multi-feature matching")
        else:
            print(f"[Identifier] No CNN model found at '{self.model_path}'")
            print(f"[Identifier] Run: python3 tools/download_model.py")
            print(f"[Identifier] Using fallback multi-feature matching")

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
            # Check if database matches current mode (CNN vs fallback)
            sample = next(iter(self.database.values()), {})
            db_has_cnn = "cnn_features" in sample
            if self.use_cnn and not db_has_cnn:
                print("[Identifier] Database was built without CNN. Rebuilding...")
                self.build_database()
            elif not self.use_cnn and db_has_cnn:
                print("[Identifier] Database was built with CNN but model not loaded. Rebuilding...")
                self.build_database()
            else:
                print(f"[Identifier] Loaded {len(self.database)} Pokémon "
                      f"({'CNN' if self.use_cnn else 'fallback'} mode)")
        elif os.path.isdir(self.ref_dir):
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            has_images = any(f.lower().endswith(exts) and not f.startswith("_")
                            for f in os.listdir(self.ref_dir))
            if has_images:
                print("[Identifier] No database found. Building...")
                self.build_database()
            else:
                print("[Identifier] No sprites found. Run:")
                print("  python3 tools/download_pokemon_sprites.py")
        else:
            print("[Identifier] No reference directory found.")

    # ------------------------------------------------------------------
    # CNN Feature Extraction
    # ------------------------------------------------------------------
    def _extract_cnn_features(self, image):
        """
        Extract a feature vector from an image using MobileNetV2.
        Returns a normalized 1000-dim vector (ImageNet logits as fingerprint).
        """
        # MobileNetV2 expects 224x224 input, normalized
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(0.485, 0.456, 0.406),  # ImageNet mean
            swapRB=False,  # Our images are already BGR from OpenCV
            crop=False
        )
        # Apply ImageNet std normalization
        # blob shape: (1, 3, 224, 224)
        blob[0, 0] /= 0.229
        blob[0, 1] /= 0.224
        blob[0, 2] /= 0.225

        self.net.setInput(blob)
        output = self.net.forward()  # Shape: (1, 1000)
        features = output.flatten()

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    # ------------------------------------------------------------------
    # Fallback Feature Extraction (no CNN)
    # ------------------------------------------------------------------
    def _extract_fallback_features(self, image):
        """
        Extract multiple feature types for matching without a CNN.
        Returns a dict of feature arrays.
        """
        img = cv2.resize(image, (200, 200))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. HSV color histogram (more bins for better discrimination)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [60], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [48], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [48], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)

        # 2. Lab color histogram (perceptually uniform color space)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hist_a = cv2.calcHist([lab], [1], None, [48], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [48], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)

        # 3. Hu moments (shape-based, rotation/scale invariant)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        # Log transform for better comparison (Hu moments span huge ranges)
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        # 4. Edge orientation histogram
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi + 180  # 0-360
        # Only count strong edges
        mask = magnitude > np.percentile(magnitude, 70)
        edge_hist, _ = np.histogram(angle[mask], bins=36, range=(0, 360))
        edge_hist = edge_hist.astype(np.float32)
        edge_norm = np.linalg.norm(edge_hist)
        if edge_norm > 0:
            edge_hist /= edge_norm

        # 5. Spatial color layout (divide into 4x4 grid, get mean color)
        grid_features = []
        gh, gw = img.shape[0] // 4, img.shape[1] // 4
        for gy in range(4):
            for gx in range(4):
                cell = hsv[gy*gh:(gy+1)*gh, gx*gw:(gx+1)*gw]
                grid_features.extend(cell.mean(axis=(0, 1)).tolist())
        grid_features = np.array(grid_features, dtype=np.float32)
        grid_norm = np.linalg.norm(grid_features)
        if grid_norm > 0:
            grid_features /= grid_norm

        return {
            "hist_h": hist_h.flatten(),
            "hist_s": hist_s.flatten(),
            "hist_v": hist_v.flatten(),
            "hist_a": hist_a.flatten(),
            "hist_b": hist_b.flatten(),
            "hu_moments": hu,
            "edge_hist": edge_hist,
            "color_grid": grid_features,
        }

    # ------------------------------------------------------------------
    # Database Building
    # ------------------------------------------------------------------
    def build_database(self):
        """Build the reference database from sprite images."""
        if not os.path.isdir(self.ref_dir):
            os.makedirs(self.ref_dir, exist_ok=True)
            return

        self.database = {}
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        filenames = sorted([f for f in os.listdir(self.ref_dir)
                           if f.lower().endswith(extensions) and not f.startswith("_")])

        if not filenames:
            print("[Identifier] No sprite images found.")
            return

        print(f"[Identifier] Processing {len(filenames)} sprites "
              f"({'CNN' if self.use_cnn else 'fallback'} mode)...\n")

        for i, filename in enumerate(filenames):
            filepath = os.path.join(self.ref_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Handle transparency
            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                bgr = img[:, :, :3]
                white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
                img_bgr = (bgr * alpha[:, :, np.newaxis] +
                           white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            elif len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img[:, :, :3]

            # Determine display name
            if filename in self.name_map:
                display_name = self.name_map[filename]
            else:
                name = os.path.splitext(filename)[0]
                display_name = name.replace("_", " ").replace("-", " ").title()
                parts = display_name.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    display_name = parts[0]

            # Extract features
            entry = {"display_name": display_name}

            if self.use_cnn:
                img_resized = cv2.resize(img_bgr, (224, 224))
                entry["cnn_features"] = self._extract_cnn_features(img_resized)
            else:
                entry["fallback_features"] = self._extract_fallback_features(img_bgr)

            self.database[filename] = entry

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(filenames)}] {display_name}")

        # Save
        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)
        print(f"\n[Identifier] Saved {len(self.database)} Pokémon to {self.db_path}")

    # ------------------------------------------------------------------
    # Card Artwork Extraction
    # ------------------------------------------------------------------
    def _extract_artwork_region(self, card_image):
        """
        Extract the artwork region from a Pokémon card.
        Tries to dynamically find the artwork box, falls back to fixed coords.
        """
        h, w = card_image.shape[:2]

        # Try to find the artwork box dynamically using edge detection
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Look for the largest internal rectangle (the artwork border)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_artwork = None
        best_area = 0
        min_area = h * w * 0.08   # At least 8% of card
        max_area = h * w * 0.65   # At most 65% of card

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, rw, rh = cv2.boundingRect(approx)
                # Artwork should be wider than tall or roughly square
                ratio = rw / max(rh, 1)
                if 0.6 < ratio < 1.8 and area > best_area:
                    best_area = area
                    best_artwork = (x, y, rw, rh)

        if best_artwork:
            x, y, rw, rh = best_artwork
            # Add small padding
            pad = 5
            x = max(0, x + pad)
            y = max(0, y + pad)
            rw = min(w - x, rw - 2 * pad)
            rh = min(h - y, rh - 2 * pad)
            return card_image[y:y+rh, x:x+rw]

        # Fallback: fixed region (works for standard Pokémon card layout)
        y1 = int(h * 0.10)
        y2 = int(h * 0.58)
        x1 = int(w * 0.06)
        x2 = int(w * 0.94)
        return card_image[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------
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

        # Extract artwork region
        artwork = self._extract_artwork_region(card_image)
        if artwork is None or artwork.size == 0:
            return {"name": "No artwork found", "confidence": 0.0, "matches": 0}

        if self.use_cnn:
            return self._identify_cnn(artwork)
        else:
            return self._identify_fallback(artwork)

    def _identify_cnn(self, artwork):
        """Identify using CNN feature cosine similarity."""
        img = cv2.resize(artwork, (224, 224))
        query_features = self._extract_cnn_features(img)

        best_name = "Unknown"
        best_score = -1
        scores = []

        for filename, ref in self.database.items():
            ref_features = ref.get("cnn_features")
            if ref_features is None:
                continue

            # Cosine similarity (both vectors are already L2-normalized)
            similarity = float(np.dot(query_features, ref_features))
            scores.append((similarity, ref["display_name"]))

            if similarity > best_score:
                best_score = similarity
                best_name = ref["display_name"]

        # Get top 3 for confidence calibration
        scores.sort(reverse=True)
        top_scores = [s[0] for s in scores[:3]]

        # Confidence: how much better is #1 than #2?
        if len(top_scores) >= 2:
            gap = top_scores[0] - top_scores[1]
            # Map the gap to confidence: 0.05 gap → ~0.6, 0.15+ gap → ~0.95
            confidence = min(0.95, 0.5 + gap * 3.0)
            confidence = max(0.1, confidence)
        else:
            confidence = max(0.1, min(0.95, best_score))

        return {
            "name": best_name,
            "confidence": float(confidence),
            "matches": int(best_score * 100),
        }

    def _identify_fallback(self, artwork):
        """Identify using multi-feature matching (no CNN)."""
        query = self._extract_fallback_features(artwork)

        best_name = "Unknown"
        best_score = -1

        for filename, ref in self.database.items():
            ref_feat = ref.get("fallback_features")
            if ref_feat is None:
                continue

            # Compare each feature type and combine
            score = 0.0

            # Color histograms (HSV) — compare with correlation
            for key in ("hist_h", "hist_s", "hist_v"):
                s = cv2.compareHist(
                    query[key].reshape(-1, 1).astype(np.float32),
                    ref_feat[key].reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                score += max(s, 0) * 0.10  # 3 histograms × 0.10 = 0.30

            # Lab histograms
            for key in ("hist_a", "hist_b"):
                s = cv2.compareHist(
                    query[key].reshape(-1, 1).astype(np.float32),
                    ref_feat[key].reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                score += max(s, 0) * 0.08  # 2 × 0.08 = 0.16

            # Hu moments (shape similarity)
            hu_dist = np.linalg.norm(query["hu_moments"] - ref_feat["hu_moments"])
            hu_score = max(0, 1.0 - hu_dist / 20.0)
            score += hu_score * 0.12

            # Edge orientation histogram
            edge_sim = float(np.dot(query["edge_hist"], ref_feat["edge_hist"]))
            score += max(edge_sim, 0) * 0.18

            # Spatial color layout
            grid_sim = float(np.dot(query["color_grid"], ref_feat["color_grid"]))
            score += max(grid_sim, 0) * 0.24

            if score > best_score:
                best_score = score
                best_name = ref["display_name"]

        confidence = float(np.clip(best_score, 0.0, 1.0))
        return {
            "name": best_name,
            "confidence": confidence,
            "matches": int(best_score * 100),
        }

    def get_card_count(self):
        """Return the number of Pokémon in the database."""
        return len(self.database)
