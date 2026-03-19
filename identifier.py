"""
Card identification via OCR.
Reads the Pokémon name (top) and card number (bottom) from a cropped card image.
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

# All Pokémon names Gen 1-2 for fuzzy matching
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
]

CARD_SUFFIXES = ["EX", "GX", "V", "VMAX", "VSTAR", "ex", "BREAK"]


class CardIdentifier:
    """Identify Pokémon cards via OCR."""

    def __init__(self):
        # Build fuzzy lookup
        self.lookup = {}
        for name in POKEMON_NAMES:
            self.lookup[name.lower()] = name
            clean = re.sub(r'[^a-z]', '', name.lower())
            self.lookup[clean] = name
        print(f"[Identifier] {len(POKEMON_NAMES)} Pokémon names loaded, "
              f"OCR {'available' if TESSERACT_AVAILABLE else 'NOT available'}")

    def identify(self, card_image):
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
        cv2.imwrite('debug_last_scan.jpg', card)
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

                    # Clean: remove numbers, slashes, special chars
                    text_alpha = re.sub(r'[^a-zA-Z.\' -]', '', text).strip()
                    all_reads.append(f"{psm[-2:]}: '{text}' -> '{text_alpha}'")

                    if len(text_alpha) < 3:
                        continue

                    name, conf = self._fuzzy_match(text_alpha)
                    if name and conf > best_conf:
                        best_name = name
                        best_conf = conf
                        best_raw = text_alpha
                except Exception:
                    continue

        # Always log everything
        print(f"[OCR name] All reads: {all_reads}")
        print(f"[OCR name] Best: '{best_name}' ({best_conf:.0%}) raw='{best_raw}'")

        # Return ALL raw reads so we can see them in the UI
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

        # Try different bottom strips
        strips = [
            card[int(h * 0.88):int(h * 0.96), int(w * 0.01):int(w * 0.55)],
            card[int(h * 0.85):int(h * 0.99), int(w * 0.01):int(w * 0.55)],
            card[int(h * 0.90):int(h * 0.99), int(w * 0.01):int(w * 0.55)],
        ]

        for region in strips:
            rh, rw = region.shape[:2]
            if rh < 3 or rw < 10:
                continue

            # Scale up a lot
            scale = max(6, 250 // max(rh, 1))
            large = cv2.resize(region, (rw * scale, rh * scale), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

            # Preprocess
            images = []
            _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            images += [t1, cv2.bitwise_not(t1)]
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            _, t2 = cv2.threshold(clahe.apply(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            images += [t2, cv2.bitwise_not(t2)]

            # NO whitelist — read everything, then search for pattern
            for img in images:
                for psm in ["--psm 6", "--psm 7", "--psm 4", "--psm 3"]:
                    try:
                        text = pytesseract.image_to_string(img, config=psm)
                        text = text.replace('\\', '/').replace('|', '/').replace('l', '1').replace('O', '0').replace('o', '0')
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
        """Create multiple preprocessed versions for OCR."""
        h, w = region.shape[:2]
        if h < 5 or w < 10:
            return []

        # Scale up (Tesseract needs ~30px font height minimum)
        scale = max(3, 100 // max(h, 1))
        large = cv2.resize(region, (w * scale, h * scale),
                           interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

        results = []

        # Otsu threshold
        _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(t1)
        results.append(cv2.bitwise_not(t1))

        # CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, t2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(t2)

        # Adaptive threshold
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
        results.append(adapt)

        return results

    # ------------------------------------------------------------------
    # Fuzzy matching
    # ------------------------------------------------------------------
    def _fuzzy_match(self, ocr_text):
        """Match OCR text against known Pokémon names."""
        text = ocr_text.strip()
        for suffix in CARD_SUFFIXES:
            text = text.replace(suffix, "").strip()
        text = re.sub(r'[^a-zA-Z.\' -]', '', text).strip()

        if len(text) < 3:
            return None, 0.0

        text_lower = text.lower()
        text_clean = re.sub(r'[^a-z]', '', text_lower)

        # Exact
        if text_lower in self.lookup:
            return self.lookup[text_lower], 0.95
        if text_clean in self.lookup:
            return self.lookup[text_clean], 0.93

        # Prefix match
        for key, name in self.lookup.items():
            if len(key) >= 4:
                if text_clean.startswith(key):
                    return name, 0.88
                if key.startswith(text_clean) and len(text_clean) >= 4:
                    return name, 0.85

        # Edit distance
        best_name, best_dist = None, 999
        for key, name in self.lookup.items():
            if abs(len(key) - len(text_clean)) > 3:
                continue
            d = self._edit_dist(text_clean, key)
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_dist <= 1 and len(text_clean) >= 4:
            return best_name, 0.85
        if best_dist <= 2 and len(text_clean) >= 5:
            return best_name, 0.70
        if best_dist <= 3 and len(text_clean) >= 6:
            return best_name, 0.55
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
        return len(POKEMON_NAMES)
