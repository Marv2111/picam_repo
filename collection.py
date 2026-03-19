"""Local card collection stored in SQLite."""

import json
import sqlite3
from datetime import datetime


class CardCollection:
    def __init__(self, db_path="collection.db"):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, card_number TEXT, grade TEXT,
                grade_score REAL, price_avg TEXT, price_low TEXT,
                price_trend TEXT, cardmarket_url TEXT, defects TEXT,
                saved_at TEXT NOT NULL
            )""")
        print(f"[Collection] DB ready: {self.db_path}")

    def save_card(self, data):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO cards (name,card_number,grade,grade_score,"
                "price_avg,price_low,price_trend,cardmarket_url,defects,saved_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (data.get("name",""), data.get("card_number",""),
                 data.get("grade",""), data.get("grade_score",0),
                 data.get("price_avg","N/A"), data.get("price_low","N/A"),
                 data.get("price_trend","N/A"), data.get("cardmarket_url",""),
                 json.dumps(data.get("defects",{})), now))
            return cur.lastrowid

    def get_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM cards ORDER BY saved_at DESC").fetchall()
        cards = []
        for r in rows:
            c = dict(r)
            try: c["defects"] = json.loads(c["defects"])
            except: c["defects"] = {}
            cards.append(c)
        return cards

    def get_total_value(self):
        cards = self.get_all()
        total, priced = 0.0, 0
        for c in cards:
            try:
                p = float(c.get("price_avg","").replace("€","").replace(",",".").strip())
                total += p; priced += 1
            except: pass
        return {"total_eur": round(total, 2), "priced_count": priced,
                "total_count": len(cards)}

    def delete_card(self, card_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cards WHERE id=?", (card_id,))

    def get_count(self):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
