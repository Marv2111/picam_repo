#!/usr/bin/env python3
"""
Build Reference Database
========================
Scans the reference_cards/ directory, computes ORB features for each image,
and saves a pickle file used by the identifier at runtime.

Usage:
    python3 tools/build_reference_db.py

Place your reference card images (JPG/PNG) in the reference_cards/ directory.
Name them descriptively, e.g.:
    pikachu_base_set_58.jpg
    charizard_base_set_4.jpg
    mewtwo_base_set_10.jpg

The filename (without extension) becomes the card's display name.
"""

import os
import sys

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identifier import CardIdentifier
import config


def main():
    ref_dir = config.REFERENCE_CARDS_DIR

    # Create reference_cards directory if it doesn't exist
    if not os.path.isdir(ref_dir):
        os.makedirs(ref_dir, exist_ok=True)
        print(f"Created '{ref_dir}/' directory.")
        print(f"\nTo get started:")
        print(f"  1. Download or photograph Pokémon card images")
        print(f"  2. Place them in the '{ref_dir}/' directory")
        print(f"  3. Re-run this script")
        print(f"\nTip: Name files descriptively, e.g.:")
        print(f"  pikachu_base_set_58.jpg")
        print(f"  charizard_base_set_4.jpg")
        return

    # Check for images
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(ref_dir) if f.lower().endswith(extensions)]

    if not images:
        print(f"No images found in '{ref_dir}/'.")
        print(f"Add card images and re-run this script.")
        return

    print(f"Found {len(images)} images in '{ref_dir}/'")
    print(f"Building ORB feature database...\n")

    # Build database
    identifier = CardIdentifier(
        db_path=config.REFERENCE_DB_PATH,
        ref_dir=ref_dir,
    )
    identifier.build_database()

    print(f"\nDone! Database saved to '{config.REFERENCE_DB_PATH}'")
    print(f"Total cards indexed: {identifier.get_card_count()}")


if __name__ == '__main__':
    main()
