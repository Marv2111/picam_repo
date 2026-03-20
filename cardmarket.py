"""Cardmarket price lookup — generates search URLs for the browser."""

import urllib.parse


class CardmarketLookup:
    """Generate Cardmarket search links (no scraping needed)."""

    def lookup(self, card_name, card_number=None):
        q = card_name
        if card_number:
            q += f" {card_number}"

        search_url = "https://www.cardmarket.com/en/Pokemon/Products/Search?" + \
                     urllib.parse.urlencode({"searchString": q})

        return {
            "found": True,
            "name": card_name,
            "price_avg": "See Cardmarket",
            "price_low": "See Cardmarket",
            "price_trend": "See Cardmarket",
            "url": search_url,
        }
