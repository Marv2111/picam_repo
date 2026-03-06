#!/usr/bin/env python3
"""
Download Sample Reference Cards
================================
Downloads Pokémon card images from the pokemontcg.io API.
Run this ONCE while connected to the internet to build your reference set.

Usage:
    python3 tools/download_sample_cards.py [--count 50]

The free API requires no key and has generous rate limits.
Images are saved to the reference_cards/ directory.
"""

import os
import sys
import time
import argparse
import urllib.request
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

API_URL = "https://api.pokemontcg.io/v2/cards"


def download_cards(count=50, output_dir=None):
    output_dir = output_dir or config.REFERENCE_CARDS_DIR
    os.makedirs(output_dir, exist_ok=True)

    page_size = min(count, 50)
    page = 1
    downloaded = 0

    print(f"Downloading up to {count} card images to '{output_dir}/'...")
    print(f"Source: pokemontcg.io API\n")

    while downloaded < count:
        url = f"{API_URL}?page={page}&pageSize={page_size}"
        print(f"Fetching page {page}...")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  API error: {e}")
            break

        cards = data.get('data', [])
        if not cards:
            print("  No more cards available.")
            break

        for card in cards:
            if downloaded >= count:
                break

            name = card.get('name', 'unknown')
            card_id = card.get('id', 'unknown')
            image_url = card.get('images', {}).get('large') or \
                        card.get('images', {}).get('small')

            if not image_url:
                continue

            # Sanitize filename
            safe_name = f"{name}_{card_id}".replace(' ', '_').replace('/', '-')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-.')
            filename = f"{safe_name}.png"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                print(f"  [SKIP] {filename} (already exists)")
                downloaded += 1
                continue

            try:
                urllib.request.urlretrieve(image_url, filepath)
                downloaded += 1
                print(f"  [{downloaded}/{count}] {filename}")
                time.sleep(0.2)  # Be polite to the API
            except Exception as e:
                print(f"  [FAIL] {filename}: {e}")

        page += 1

    print(f"\nDownloaded {downloaded} card images.")
    print(f"Now run: python3 tools/build_reference_db.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download sample card images')
    parser.add_argument('--count', type=int, default=50,
                        help='Number of cards to download')
    args = parser.parse_args()
    download_cards(args.count)
