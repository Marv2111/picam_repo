#!/usr/bin/env python3
"""
Download Pokémon Artwork Sprites
=================================
Downloads official Pokémon artwork from PokeAPI (100% free, no key needed).
These are used to identify which Pokémon appears on a card.

Usage:
    python3 tools/download_pokemon_sprites.py [--count 151]
"""

import os
import sys
import json
import time
import argparse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTWORK_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{}.png"
POKEAPI_URL = "https://pokeapi.co/api/v2/pokemon/{}"
OUTPUT_DIR = "reference_cards"


def get_pokemon_name(poke_id):
    """Fetch the English name for a Pokémon ID from PokeAPI."""
    url = f"https://pokeapi.co/api/v2/pokemon-species/{poke_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            # Get English name
            for entry in data.get("names", []):
                if entry.get("language", {}).get("name") == "en":
                    return entry["name"]
            return data.get("name", f"pokemon_{poke_id}").capitalize()
    except Exception:
        return f"Pokemon_{poke_id}"


def main():
    parser = argparse.ArgumentParser(description="Download Pokémon artwork sprites")
    parser.add_argument("--count", type=int, default=151,
                        help="Number of Pokémon to download (default: 151 = Gen 1)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Also save a name mapping file
    name_map = {}

    print(f"Downloading artwork for {args.count} Pokémon...")
    print(f"Source: PokeAPI (free)\n")

    downloaded = 0
    for poke_id in range(1, args.count + 1):
        # Get the proper name
        name = get_pokemon_name(poke_id)
        safe_name = name.replace(" ", "_").replace("'", "").replace(".", "")
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')

        filename = f"{safe_name}_{poke_id}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            print(f"  [SKIP] {filename} (exists)")
            name_map[filename] = name
            downloaded += 1
            continue

        # Download artwork
        img_url = ARTWORK_URL.format(poke_id)
        req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                with open(filepath, "wb") as f:
                    f.write(r.read())
            name_map[filename] = name
            downloaded += 1
            print(f"  [{downloaded}/{args.count}] {name} -> {filename}")
            time.sleep(0.15)
        except Exception as e:
            print(f"  [FAIL] {name} ({poke_id}): {e}")

    # Save name mapping
    map_path = os.path.join(OUTPUT_DIR, "_name_map.json")
    with open(map_path, "w") as f:
        json.dump(name_map, f, indent=2)

    print(f"\nDownloaded {downloaded} Pokémon sprites to '{OUTPUT_DIR}/'")
    print(f"Name mapping saved to '{map_path}'")
    print(f"\nNext: python3 tools/build_reference_db.py")


if __name__ == "__main__":
    main()
