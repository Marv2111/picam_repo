#!/usr/bin/env python3
"""
Build Reference Database
========================
Builds the Pokémon feature database from sprite images.
Uses CNN features (MobileNetV2) if the model is available,
otherwise uses multi-feature fallback.

Usage:
    python3 tools/build_reference_db.py

Prerequisites:
    python3 tools/download_pokemon_sprites.py   (get sprite images)
    python3 tools/download_model.py              (get CNN model — recommended)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identifier import CardIdentifier
import config


def main():
    ref_dir = config.REFERENCE_CARDS_DIR
    model_path = os.path.join("models", "mobilenetv2-12.onnx")

    # Check prerequisites
    if not os.path.isdir(ref_dir):
        os.makedirs(ref_dir, exist_ok=True)
        print(f"Created '{ref_dir}/' directory.")
        print(f"Run: python3 tools/download_pokemon_sprites.py")
        return

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [f for f in os.listdir(ref_dir)
              if f.lower().endswith(extensions) and not f.startswith("_")]

    if not images:
        print(f"No images in '{ref_dir}/'.")
        print(f"Run: python3 tools/download_pokemon_sprites.py")
        return

    print(f"Found {len(images)} sprite images")

    if os.path.exists(model_path):
        print(f"CNN model found — using MobileNetV2 features (best accuracy)")
    else:
        print(f"No CNN model — using fallback features")
        print(f"For better accuracy: python3 tools/download_model.py\n")

    # Delete old database to force rebuild
    if os.path.exists(config.REFERENCE_DB_PATH):
        os.remove(config.REFERENCE_DB_PATH)

    # Build
    identifier = CardIdentifier(
        db_path=config.REFERENCE_DB_PATH,
        ref_dir=ref_dir,
        model_path=model_path,
    )
    identifier.build_database()

    print(f"\nDone! {identifier.get_card_count()} Pokémon indexed.")


if __name__ == "__main__":
    main()
