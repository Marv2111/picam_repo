#!/usr/bin/env python3
"""
Build Reference Database
========================
Scans the reference_cards/ directory for Pokémon sprite images,
computes color histograms and ORB features, and saves a database file.

Usage:
    python3 tools/build_reference_db.py

First download sprites:
    python3 tools/download_pokemon_sprites.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identifier import CardIdentifier
import config


def main():
    ref_dir = config.REFERENCE_CARDS_DIR

    if not os.path.isdir(ref_dir):
        os.makedirs(ref_dir, exist_ok=True)
        print(f"Created '{ref_dir}/' directory.")
        print(f"\nDownload sprites first:")
        print(f"  python3 tools/download_pokemon_sprites.py")
        return

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [f for f in os.listdir(ref_dir)
              if f.lower().endswith(extensions) and not f.startswith("_")]

    if not images:
        print(f"No images found in '{ref_dir}/'.")
        print(f"Run: python3 tools/download_pokemon_sprites.py")
        return

    print(f"Found {len(images)} sprite images in '{ref_dir}/'")
    print(f"Building feature database...\n")

    identifier = CardIdentifier(
        db_path=config.REFERENCE_DB_PATH,
        ref_dir=ref_dir,
    )
    identifier.build_database()

    print(f"\nDone! Database: '{config.REFERENCE_DB_PATH}'")
    print(f"Total Pokémon indexed: {identifier.get_card_count()}")


if __name__ == "__main__":
    main()
