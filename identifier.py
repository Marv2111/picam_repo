"""
Pokémon identification module — OCR-based.
Reads the Pokémon name printed on the card using Tesseract OCR,
then fuzzy-matches it against a list of all known Pokémon names.

This is far more accurate than image matching because every Pokémon card
has the name clearly printed at the top.

Fallback: If OCR fails, uses CNN feature matching against sprites.
"""

import os
import json
import pickle
import re
import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

import config

# All 151 Gen 1 Pokémon names for fuzzy matching
# Extend this list for more generations
POKEMON_NAMES = [
    "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon",
    "Charizard", "Squirtle", "Wartortle", "Blastoise", "Caterpie",
    "Metapod", "Butterfree", "Weedle", "Kakuna", "Beedrill",
    "Pidgey", "Pidgeotto", "Pidgeot", "Rattata", "Raticate",
    "Spearow", "Fearow", "Ekans", "Arbok", "Pikachu",
    "Raichu", "Sandshrew", "Sandslash", "Nidoran", "Nidorina",
    "Nidoqueen", "Nidorino", "Nidoking", "Clefairy", "Clefable",
    "Vulpix", "Ninetales", "Jigglypuff", "Wigglytuff", "Zubat",
    "Golbat", "Oddish", "Gloom", "Vileplume", "Paras",
    "Parasect", "Venonat", "Venomoth", "Diglett", "Dugtrio",
    "Meowth", "Persian", "Psyduck", "Golduck", "Mankey",
    "Primeape", "Growlithe", "Arcanine", "Poliwag", "Poliwhirl",
    "Poliwrath", "Abra", "Kadabra", "Alakazam", "Machop",
    "Machoke", "Machamp", "Bellsprout", "Weepinbell", "Victreebel",
    "Tentacool", "Tentacruel", "Geodude", "Graveler", "Golem",
    "Ponyta", "Rapidash", "Slowpoke", "Slowbro", "Magnemite",
    "Magneton", "Farfetchd", "Doduo", "Dodrio", "Seel",
    "Dewgong", "Grimer", "Muk", "Shellder", "Cloyster",
    "Gastly", "Haunter", "Gengar", "Onix", "Drowzee",
    "Hypno", "Krabby", "Kingler", "Voltorb", "Electrode",
    "Exeggcute", "Exeggutor", "Cubone", "Marowak", "Hitmonlee",
    "Hitmonchan", "Lickitung", "Koffing", "Weezing", "Rhyhorn",
    "Rhydon", "Chansey", "Tangela", "Kangaskhan", "Horsea",
    "Seadra", "Goldeen", "Seaking", "Staryu", "Starmie",
    "Mr. Mime", "Scyther", "Jynx", "Electabuzz", "Magmar",
    "Pinsir", "Tauros", "Magikarp", "Gyarados", "Lapras",
    "Ditto", "Eevee", "Vaporeon", "Jolteon", "Flareon",
    "Porygon", "Omanyte", "Omastar", "Kabuto", "Kabutops",
    "Aerodactyl", "Snorlax", "Articuno", "Zapdos", "Moltres",
    "Dratini", "Dragonair", "Dragonite", "Mewtwo", "Mew",
    # Gen 2
    "Chikorita", "Bayleef", "Meganium", "Cyndaquil", "Quilava",
    "Typhlosion", "Totodile", "Croconaw", "Feraligatr", "Sentret",
    "Furret", "Hoothoot", "Noctowl", "Ledyba", "Ledian",
    "Spinarak", "Ariados", "Crobat", "Chinchou", "Lanturn",
    "Pichu", "Cleffa", "Igglybuff", "Togepi", "Togetic",
    "Natu", "Xatu", "Mareep", "Flaaffy", "Ampharos",
    "Bellossom", "Marill", "Azumarill", "Sudowoodo", "Politoed",
    "Hoppip", "Skiploom", "Jumpluff", "Aipom", "Sunkern",
    "Sunflora", "Yanma", "Wooper", "Quagsire", "Espeon",
    "Umbreon", "Murkrow", "Slowking", "Misdreavus", "Unown",
    "Wobbuffet", "Girafarig", "Pineco", "Forretress", "Dunsparce",
    "Gligar", "Steelix", "Snubbull", "Granbull", "Qwilfish",
    "Scizor", "Shuckle", "Heracross", "Sneasel", "Teddiursa",
    "Ursaring", "Slugma", "Magcargo", "Swinub", "Piloswine",
    "Corsola", "Remoraid", "Octillery", "Delibird", "Mantine",
    "Skarmory", "Houndour", "Houndoom", "Kingdra", "Phanpy",
    "Donphan", "Porygon2", "Stantler", "Smeargle", "Tyrogue",
    "Hitmontop", "Smoochum", "Elekid", "Magby", "Miltank",
    "Blissey", "Raikou", "Entei", "Suicune", "Larvitar",
    "Pupitar", "Tyranitar", "Lugia", "Ho-Oh", "Celebi",
    # Common card suffixes/prefixes to handle
    # (these help match "Pikachu V", "Charizard EX", "Mewtwo GX" etc.)
]

# Common suffixes on modern cards
CARD_SUFFIXES = ["EX", "GX", "V", "VMAX", "VSTAR", "ex", "LV.X",
                 "BREAK", "LEGEND", "Lv.X", "Star", "δ"]


class CardIdentifier:
    """Identify Pokémon cards using OCR to read the name."""

    def __init__(self, db_path=None, ref_dir=None, model_path=None):
        self.db_path = db_path or config.REFERENCE_DB_PATH
        self.ref_dir = ref_dir or config.REFERENCE_CARDS_DIR

        # Build lowercase lookup for fuzzy matching
        self.pokemon_lookup = {}
        for name in POKEMON_NAMES:
            self.pokemon_lookup[name.lower()] = name
            # Also index without special characters
            clean = re.sub(r'[^a-z]', '', name.lower())
            self.pokemon_lookup[clean] = name

        # CNN fallback
        self.net = None
        self.use_cnn = False
        self.database = {}
        model_path = model_path or os.path.join("models", "mobilenetv2-12.onnx")
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.use_cnn = True
            except Exception:
                pass

        self.name_map = {}
        map_path = os.path.join(self.ref_dir, "_name_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                self.name_map = json.load(f)

        self._load_database()

        if TESSERACT_AVAILABLE:
            print("[Identifier] OCR mode (Tesseract) — best accuracy")
        else:
            print("[Identifier] WARNING: pytesseract not installed!")
            print("  Install with: sudo apt install tesseract-ocr && pip install pytesseract")
            if self.use_cnn:
                print("[Identifier] Falling back to CNN matching")
            else:
                print("[Identifier] No identification method available!")

    def _load_database(self):
        """Load CNN feature database if available."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.database = pickle.load(f)
            print(f"[Identifier] Loaded {len(self.database)} sprites for fallback")

    def build_database(self):
        """Build CNN feature database from sprites (for fallback matching)."""
        if not self.use_cnn or not os.path.isdir(self.ref_dir):
            print("[Identifier] Skipping CNN database (no model or no sprites)")
            return

        self.database = {}
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        filenames = sorted([f for f in os.listdir(self.ref_dir)
                           if f.lower().endswith(extensions) and not f.startswith("_")])

        print(f"[Identifier] Building CNN fallback database ({len(filenames)} sprites)...")

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

            # Get name
            if filename in self.name_map:
                display_name = self.name_map[filename]
            else:
                name = os.path.splitext(filename)[0]
                display_name = name.replace("_", " ").replace("-", " ").title()
                parts = display_name.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    display_name = parts[0]

            # CNN features
            img_resized = cv2.resize(img_bgr, (224, 224))
            features = self._extract_cnn_features(img_resized)

            self.database[filename] = {
                "display_name": display_name,
                "cnn_features": features,
            }

            if (i + 1) % 30 == 0:
                print(f"  [{i+1}/{len(filenames)}]")

        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)
        print(f"[Identifier] Saved {len(self.database)} entries")

    def _extract_cnn_features(self, image):
        """Extract CNN feature vector from an image."""
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0, size=(224, 224),
            mean=(0.485, 0.456, 0.406), swapRB=False, crop=False
        )
        blob[0, 0] /= 0.229
        blob[0, 1] /= 0.224
        blob[0, 2] /= 0.225
        self.net.setInput(blob)
        output = self.net.forward().flatten()
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm
        return output

    # ------------------------------------------------------------------
    # OCR-based identification (primary method)
    # ------------------------------------------------------------------
    def _extract_name_region(self, card_image):
        """
        Extract the top region of the card where the Pokémon name is printed.
        On standard cards, the name is in roughly the top 12-15% of the card.
        """
        h, w = card_image.shape[:2]

        # Name region: top portion of card, slightly inset from edges
        y1 = int(h * 0.02)
        y2 = int(h * 0.14)
        x1 = int(w * 0.05)
        x2 = int(w * 0.80)  # Don't go full width (HP value is on the right)

        return card_image[y1:y2, x1:x2]

    def _preprocess_for_ocr(self, region):
        """
        Preprocess an image region for better OCR accuracy.
        Returns multiple preprocessed versions to try.
        """
        results = []

        # Scale up for better OCR (Tesseract likes larger text)
        h, w = region.shape[:2]
        scale = max(3, 150 // max(h, 1))
        large = cv2.resize(region, (w * scale, h * scale),
                           interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

        # Version 1: Simple threshold
        _, thresh1 = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(thresh1)

        # Version 2: Inverted threshold (for dark backgrounds)
        results.append(cv2.bitwise_not(thresh1))

        # Version 3: Adaptive threshold
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
        results.append(adapt)

        # Version 4: High contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh4 = cv2.threshold(enhanced, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(thresh4)

        return results

    def _ocr_read(self, image_variants):
        """
        Run Tesseract OCR on multiple image variants, return all detected text.
        """
        if not TESSERACT_AVAILABLE:
            return []

        texts = []
        ocr_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.' -"

        for img in image_variants:
            try:
                text = pytesseract.image_to_string(img, config=ocr_config).strip()
                if text and len(text) >= 3:
                    texts.append(text)
            except Exception:
                continue

        return texts

    def _fuzzy_match(self, ocr_text):
        """
        Match OCR text against known Pokémon names.
        Returns (name, confidence) or (None, 0).
        """
        if not ocr_text:
            return None, 0.0

        # Clean the OCR text
        text = ocr_text.strip()

        # Remove common card suffixes
        for suffix in CARD_SUFFIXES:
            text = text.replace(suffix, "").strip()

        # Remove trailing/leading junk
        text = re.sub(r'[^a-zA-Z.\' -]', '', text).strip()

        if len(text) < 3:
            return None, 0.0

        text_lower = text.lower()
        text_clean = re.sub(r'[^a-z]', '', text_lower)

        # Exact match
        if text_lower in self.pokemon_lookup:
            return self.pokemon_lookup[text_lower], 0.95

        if text_clean in self.pokemon_lookup:
            return self.pokemon_lookup[text_clean], 0.93

        # Starts-with match (handles OCR reading extra chars)
        for key, name in self.pokemon_lookup.items():
            if text_clean.startswith(key) and len(key) >= 4:
                return name, 0.88
            if key.startswith(text_clean) and len(text_clean) >= 4:
                return name, 0.85

        # Levenshtein-like distance (simple edit distance)
        best_name = None
        best_dist = 999
        for key, name in self.pokemon_lookup.items():
            if abs(len(key) - len(text_clean)) > 3:
                continue
            dist = self._edit_distance(text_clean, key)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist <= 1 and len(text_clean) >= 4:
            return best_name, 0.85
        elif best_dist <= 2 and len(text_clean) >= 5:
            return best_name, 0.70
        elif best_dist <= 3 and len(text_clean) >= 6:
            return best_name, 0.55

        return None, 0.0

    def _edit_distance(self, s1, s2):
        """Simple Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    # ------------------------------------------------------------------
    # CNN fallback identification
    # ------------------------------------------------------------------
    def _identify_cnn(self, card_image):
        """Identify using CNN feature matching (fallback)."""
        if not self.use_cnn or not self.database:
            return {"name": "Unknown", "confidence": 0.0, "matches": 0}

        h, w = card_image.shape[:2]
        artwork = card_image[int(h*0.10):int(h*0.58), int(w*0.06):int(w*0.94)]
        img = cv2.resize(artwork, (224, 224))
        query = self._extract_cnn_features(img)

        best_name = "Unknown"
        best_score = -1

        for filename, ref in self.database.items():
            ref_feat = ref.get("cnn_features")
            if ref_feat is None:
                continue
            similarity = float(np.dot(query, ref_feat))
            if similarity > best_score:
                best_score = similarity
                best_name = ref["display_name"]

        confidence = float(np.clip(best_score * 0.8, 0.0, 0.7))
        return {
            "name": best_name,
            "confidence": confidence,
            "matches": int(best_score * 100),
        }

    # ------------------------------------------------------------------
    # Card number reading (e.g., "198/217")
    # ------------------------------------------------------------------
    def _extract_number_region(self, card_image):
        """
        Extract the bottom-left region where the card number is printed.
        Standard cards show the number like "198/217" at the bottom.
        """
        h, w = card_image.shape[:2]
        # Card number is typically bottom-left area
        y1 = int(h * 0.90)
        y2 = int(h * 0.98)
        x1 = int(w * 0.03)
        x2 = int(w * 0.45)
        return card_image[y1:y2, x1:x2]

    def _read_card_number(self, card_image):
        """
        Read the card number (e.g., "198/217") from the card.
        Returns the number string or empty string if not found.
        """
        if not TESSERACT_AVAILABLE:
            return ""

        region = self._extract_number_region(card_image)
        h, w = region.shape[:2]
        if h < 5 or w < 10:
            return ""

        # Scale up
        scale = max(3, 120 // max(h, 1))
        large = cv2.resize(region, (w * scale, h * scale),
                           interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

        # Try multiple preprocessing
        variants = []
        _, thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh)
        variants.append(cv2.bitwise_not(thresh))

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh2 = cv2.threshold(enhanced, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh2)

        # OCR with digits and slash allowed
        ocr_config = "--psm 7 -c tessedit_char_whitelist=0123456789/"

        for img in variants:
            try:
                text = pytesseract.image_to_string(img, config=ocr_config).strip()
                # Look for pattern like "198/217"
                match = re.search(r'(\d{1,4})\s*/\s*(\d{1,4})', text)
                if match:
                    return f"{match.group(1)}/{match.group(2)}"
            except Exception:
                continue

        return ""

    # ------------------------------------------------------------------
    # Main identification entry point
    # ------------------------------------------------------------------
    def identify(self, card_image):
        """
        Identify which Pokémon is on the card.
        Primary: OCR (reads the name text)
        Fallback: CNN feature matching
        Also reads the card number (e.g., "198/217").

        Args:
            card_image: BGR image of a cropped, perspective-corrected card.

        Returns:
            dict with 'name', 'confidence', 'matches', 'card_number', etc.
        """
        # Read card number
        card_number = self._read_card_number(card_image)

        # Try OCR first for the name
        if TESSERACT_AVAILABLE:
            name_region = self._extract_name_region(card_image)
            image_variants = self._preprocess_for_ocr(name_region)
            ocr_texts = self._ocr_read(image_variants)

            # Try matching each OCR result
            best_name = None
            best_conf = 0.0
            best_text = ""

            for text in ocr_texts:
                name, conf = self._fuzzy_match(text)
                if name and conf > best_conf:
                    best_name = name
                    best_conf = conf
                    best_text = text

            if best_name and best_conf >= 0.55:
                return {
                    "name": best_name,
                    "confidence": best_conf,
                    "matches": 0,
                    "method": "OCR",
                    "raw_ocr": best_text,
                    "card_number": card_number,
                }

        # Fallback to CNN
        cnn_result = self._identify_cnn(card_image)
        cnn_result["method"] = "CNN (fallback)"
        cnn_result["card_number"] = card_number
        return cnn_result

    def get_card_count(self):
        """Return the number of Pokémon in the database."""
        return max(len(self.database), len(POKEMON_NAMES))
