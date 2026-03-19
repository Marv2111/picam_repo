"""Cardmarket price lookup via GET requests only."""

import re
import json
import time
import urllib.request
import urllib.parse


class CardmarketLookup:
    """Look up Pokémon card prices on Cardmarket."""

    SEARCH_URL = "https://www.cardmarket.com/en/Pokemon/Products/Search"
    BASE_URL = "https://www.cardmarket.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    }

    def __init__(self):
        self._cache = {}
        self._last_req = 0

    def lookup(self, card_name, card_number=None):
        key = f"{card_name}_{card_number or ''}"
        if key in self._cache:
            return self._cache[key]

        # Rate limit
        wait = 2.0 - (time.time() - self._last_req)
        if wait > 0:
            time.sleep(wait)

        result = self._search(card_name, card_number)
        self._cache[key] = result
        return result

    def _search(self, name, number):
        try:
            q = name + (f" {number}" if number else "")
            params = urllib.parse.urlencode({"searchString": q})
            url = f"{self.SEARCH_URL}?{params}"
            req = urllib.request.Request(url, headers=self.HEADERS)
            self._last_req = time.time()

            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                final_url = resp.url

            if "/Products/Singles/" in final_url:
                return self._parse_product(html, final_url)

            return self._parse_results(html, name)

        except Exception as e:
            return {"found": False, "error": str(e)}

    def _parse_results(self, html, name):
        links = re.findall(r'href="(/en/Pokemon/Products/Singles/[^"]+)"', html)
        if not links:
            return {"found": False, "error": "No results found"}

        name_slug = name.lower().replace(" ", "-")
        best = None
        for link in links:
            if name_slug in link.lower():
                best = link
                break
        if not best:
            best = links[0]

        try:
            url = f"{self.BASE_URL}{best}"
            req = urllib.request.Request(url, headers=self.HEADERS)
            time.sleep(1)
            self._last_req = time.time()
            with urllib.request.urlopen(req, timeout=15) as resp:
                return self._parse_product(resp.read().decode("utf-8", errors="ignore"), url)
        except Exception as e:
            return {"found": True, "name": name, "price_avg": "N/A",
                    "price_low": "N/A", "price_trend": "N/A",
                    "url": f"{self.BASE_URL}{best}", "error": str(e)}

    def _parse_product(self, html, url):
        r = {"found": True, "name": "", "price_avg": "N/A",
             "price_low": "N/A", "price_trend": "N/A", "url": url}

        m = re.search(r'<title>([^<]+)</title>', html)
        if m:
            r["name"] = re.sub(r'\s*[|\-].*$', '', m.group(1)).strip()

        for pat in [r'Price Trend[^€]*?(\d+[.,]\d{2})\s*€',
                    r'Preistrend[^€]*?(\d+[.,]\d{2})\s*€',
                    r'"avg"[^}]*?(\d+\.?\d*)']:
            m = re.search(pat, html, re.I)
            if m:
                r["price_trend"] = r["price_avg"] = f"€{m.group(1).replace(',','.')}"
                break

        for pat in [r'From[^€]*?(\d+[.,]\d{2})\s*€',
                    r'Ab\s*(\d+[.,]\d{2})\s*€']:
            m = re.search(pat, html, re.I)
            if m:
                r["price_low"] = f"€{m.group(1).replace(',','.')}"
                break

        # Try JSON-LD
        jm = re.search(r'type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S)
        if jm:
            try:
                ld = json.loads(jm.group(1))
                offers = ld.get("offers", {})
                if offers.get("lowPrice"):
                    r["price_low"] = f"€{offers['lowPrice']}"
                if offers.get("price") and r["price_avg"] == "N/A":
                    r["price_avg"] = f"€{offers['price']}"
                if not r["name"] and ld.get("name"):
                    r["name"] = ld["name"]
            except Exception:
                pass

        return r
