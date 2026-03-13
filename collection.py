"""
Card collection database.
Stores saved cards with name, number, grade, price, and timestamp
in a local SQLite database.
"""

import os
import json
import sqlite3
import time
from datetime import datetime


DB_PATH = "collection.db"


class CardCollection:
    """Local card collection stored in SQLite."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        """Create the database table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    card_number TEXT,
                    set_name TEXT,
                    grade TEXT,
                    grade_score REAL,
                    price_avg TEXT,
                    price_low TEXT,
                    price_trend TEXT,
                    cardmarket_url TEXT,
                    defects TEXT,
                    saved_at TEXT NOT NULL
                )
            """)
            conn.commit()
        print(f"[Collection] Database ready: {self.db_path}")

    def save_card(self, card_data):
        """
        Save a card to the collection.

        Args:
            card_data: dict with keys like name, card_number, grade, price, etc.

        Returns:
            int: the row ID of the saved card
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        defects_json = json.dumps(card_data.get("defects", {}))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO cards (
                    name, card_number, set_name, grade, grade_score,
                    price_avg, price_low, price_trend, cardmarket_url,
                    defects, saved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card_data.get("name", "Unknown"),
                card_data.get("card_number", ""),
                card_data.get("set_name", ""),
                card_data.get("grade", ""),
                card_data.get("grade_score", 0.0),
                card_data.get("price_avg", "N/A"),
                card_data.get("price_low", "N/A"),
                card_data.get("price_trend", "N/A"),
                card_data.get("cardmarket_url", ""),
                defects_json,
                now,
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all(self):
        """Get all saved cards, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM cards ORDER BY saved_at DESC"
            ).fetchall()

        cards = []
        for row in rows:
            card = dict(row)
            try:
                card["defects"] = json.loads(card["defects"])
            except (json.JSONDecodeError, TypeError):
                card["defects"] = {}
            cards.append(card)

        return cards

    def get_total_value(self):
        """Calculate total estimated collection value."""
        cards = self.get_all()
        total = 0.0
        count = 0
        for card in cards:
            price_str = card.get("price_avg", "N/A")
            try:
                price = float(price_str.replace("€", "").replace(",", ".").strip())
                total += price
                count += 1
            except (ValueError, AttributeError):
                pass

        return {"total_eur": round(total, 2), "priced_count": count,
                "total_count": len(cards)}

    def delete_card(self, card_id):
        """Delete a card from the collection by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            conn.commit()

    def get_count(self):
        """Get the number of cards in the collection."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        return count
