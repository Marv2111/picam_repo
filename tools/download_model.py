#!/usr/bin/env python3
"""
Download MobileNetV2 ONNX Model
================================
Downloads a pre-trained MobileNetV2 model for Pokémon identification.
Uses the ONNX Model Zoo (free, no API key needed).

The model is used as a feature extractor — we compare the "fingerprint"
of the card artwork against stored Pokémon sprite fingerprints.

Usage:
    python3 tools/download_model.py
"""

import os
import sys
import urllib.request

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2-12.onnx")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        return

    print(f"Downloading MobileNetV2 ONNX model...")
    print(f"Source: ONNX Model Zoo (GitHub)")
    print(f"This is ~14 MB, may take a minute...\n")

    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})

        # Download with progress
        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536

            with open(MODEL_PATH, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  {mb:.1f} MB / {total / (1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)

        print(f"\n\nSaved to: {MODEL_PATH}")
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
        print(f"\nNext: python3 tools/build_reference_db.py")

    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print("Check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
