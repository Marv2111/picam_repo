"""
Cardmarket price lookup module.
Searches Cardmarket.com for Pokémon card prices using GET requests only.
No data is sent — only public page reads.
"""

import re
import urllib.request
import urllib.parse
import json
import time


class CardmarketLookup:
    """Look up Pokémon card prices on Cardmarket."""

    BASE_URL = "https://www.cardmarket.com"
    SEARCH_URL = "https://www.cardmarket.com/en/Pokemon/Products/Search"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self):
        self._cache = {}  # Simple in-memory cache
        self._last_request = 0
        self._rate_limit = 2.0  # Seconds between requests

    def lookup(self, card_name, card_number=None):
        """
        Search Cardmarket for a card and return price info.

        Args:
            card_name: Pokémon name (e.g., "Pikachu")
            card_number: Card number if available (e.g., "198/217")

        Returns:
            dict with:
                - 'found': bool
                - 'name': full card name from Cardmarket
                - 'price_avg': average price in EUR (str)
                - 'price_low': lowest price in EUR (str)
                - 'url': link to the card page
                - 'error': error message if lookup failed
        """
        # Build cache key
        cache_key = f"{card_name}_{card_number or 'no_num'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Rate limit
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

        # Build search query
        search_terms = card_name
        if card_number:
            search_terms += f" {card_number}"

        result = self._search_card(search_terms, card_name, card_number)
        self._cache[cache_key] = result
        return result

    def _search_card(self, search_terms, card_name, card_number):
        """Perform the Cardmarket search."""
        try:
            # URL encode the search
            params = urllib.parse.urlencode({"searchString": search_terms})
            url = f"{self.SEARCH_URL}?{params}"

            req = urllib.request.Request(url, headers=self.HEADERS)
            self._last_request = time.time()

            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Check if we landed on a product page directly
            if "/Products/Singles/" in resp.url:
                return self._parse_product_page(html, resp.url)

            # Otherwise parse search results
            return self._parse_search_results(html, card_name, card_number)

        except urllib.error.HTTPError as e:
            return {"found": False, "error": f"HTTP {e.code}"}
        except urllib.error.URLError as e:
            return {"found": False, "error": f"Connection error: {e.reason}"}
        except Exception as e:
            return {"found": False, "error": str(e)}

    def _parse_search_results(self, html, card_name, card_number):
        """Parse search results page to find the card."""
        # Look for product links with prices
        # Cardmarket search results contain product tiles with price info
        
        # Find product links
        product_pattern = r'<a[^>]*href="(/en/Pokemon/Products/Singles/[^"]+)"[^>]*>'
        product_links = re.findall(product_pattern, html)

        if not product_links:
            return {"found": False, "error": "No results found on Cardmarket"}

        # Try to find the best matching product
        best_link = None
        name_lower = card_name.lower().replace(" ", "-").replace(".", "")

        for link in product_links:
            link_lower = link.lower()
            if name_lower in link_lower:
                # If we have a card number, try to match it
                if card_number:
                    num_clean = card_number.replace("/", "-").replace(" ", "")
                    if num_clean.lower() in link_lower:
                        best_link = link
                        break
                if best_link is None:
                    best_link = link

        if not best_link and product_links:
            best_link = product_links[0]

        if not best_link:
            return {"found": False, "error": "Could not match card in results"}

        # Fetch the product page
        try:
            product_url = f"{self.BASE_URL}{best_link}"
            req = urllib.request.Request(product_url, headers=self.HEADERS)
            time.sleep(1)  # Be polite
            self._last_request = time.time()

            with urllib.request.urlopen(req, timeout=15) as resp:
                product_html = resp.read().decode("utf-8", errors="ignore")

            return self._parse_product_page(product_html, product_url)

        except Exception as e:
            # Return at least the link even if we can't get price
            return {
                "found": True,
                "name": card_name,
                "price_avg": "N/A",
                "price_low": "N/A",
                "url": f"{self.BASE_URL}{best_link}",
                "error": f"Could not load product page: {e}",
            }

    def _parse_product_page(self, html, url):
        """Parse a Cardmarket product page for price info."""
        result = {
            "found": True,
            "name": "",
            "price_avg": "N/A",
            "price_low": "N/A",
            "price_trend": "N/A",
            "url": url,
        }

        # Extract card name from page title
        title_match = re.search(r'<title>([^<]+)</title>', html)
        if title_match:
            title = title_match.group(1)
            # Clean up title (remove " | Cardmarket" suffix etc.)
            title = re.sub(r'\s*\|.*$', '', title).strip()
            title = re.sub(r'\s*-\s*Cardmarket.*$', '', title).strip()
            result["name"] = title

        # Look for price trend / average price
        # Cardmarket uses various formats for prices

        # Pattern: "Price Trend" or "Avg. Sell Price" followed by a price
        trend_patterns = [
            r'Price Trend[^€$]*?(\d+[.,]\d{2})\s*€',
            r'Preistrend[^€$]*?(\d+[.,]\d{2})\s*€',
            r'price-trend[^€$]*?(\d+[.,]\d{2})\s*€',
            r'"priceGuide"[^}]*?"avg"[^}]*?(\d+\.?\d*)',
            r'Average[^€$]*?(\d+[.,]\d{2})\s*€',
            r'avg["\s:]+(\d+\.?\d*)',
        ]

        for pattern in trend_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                price = match.group(1).replace(",", ".")
                result["price_trend"] = f"€{price}"
                result["price_avg"] = f"€{price}"
                break

        # Look for lowest price ("From" price)
        low_patterns = [
            r'From[^€$]*?(\d+[.,]\d{2})\s*€',
            r'Ab\s*(\d+[.,]\d{2})\s*€',
            r'"lowPrice"[^}]*?(\d+\.?\d*)',
            r'low["\s:]+(\d+\.?\d*)',
        ]

        for pattern in low_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                price = match.group(1).replace(",", ".")
                result["price_low"] = f"€{price}"
                break

        # Try JSON-LD structured data (some pages have this)
        json_ld_match = re.search(
            r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
            html, re.DOTALL
        )
        if json_ld_match:
            try:
                ld_data = json.loads(json_ld_match.group(1))
                if isinstance(ld_data, dict):
                    offers = ld_data.get("offers", {})
                    if isinstance(offers, dict):
                        low = offers.get("lowPrice")
                        if low:
                            result["price_low"] = f"€{low}"
                        price = offers.get("price")
                        if price and result["price_avg"] == "N/A":
                            result["price_avg"] = f"€{price}"
                    if not result["name"] and ld_data.get("name"):
                        result["name"] = ld_data["name"]
            except (json.JSONDecodeError, KeyError):
                pass

        return result

    def clear_cache(self):
        """Clear the price cache."""
        self._cache = {}
