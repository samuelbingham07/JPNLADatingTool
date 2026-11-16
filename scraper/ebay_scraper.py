"""
eBay scraper using the Browse API (OAuth client credentials).

Browse API returns active listings with full-size images.
No Playwright required — pure REST.

Usage:
    python -m scraper.ebay_scraper
    python -m scraper.ebay_scraper --query "vintage levi's 501" --pages 3
    python -m scraper.ebay_scraper --all-queries --pages 5
    python -m scraper.ebay_scraper --no-label-filter   # all listings, not just dated ones
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from scraper.config import (
    DATA_DIR, EBAY_APP_ID, EBAY_CERT_ID,
    DELAY_BETWEEN_REQUESTS, MAX_IMAGES_PER_LISTING,
    MAX_PAGES_PER_QUERY, SEARCH_QUERIES,
)
from scraper.label_extractor import extract as extract_label

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
PAGE_SIZE = 100  # Browse API max per request


class EbayClient:
    def __init__(self, client: httpx.AsyncClient):
        self._http = client
        self._token: str | None = None
        self._token_expires: float = 0

    async def _ensure_token(self) -> None:
        if self._token and time.time() < self._token_expires - 60:
            return
        import base64
        creds = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
        r = await self._http.post(
            TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {creds}",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 7200)

    async def search(self, query: str, offset: int = 0) -> dict:
        await self._ensure_token()
        r = await self._http.get(
            BROWSE_URL,
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
            },
            params={
                "q": query,
                "category_ids": "11450",       # Clothing, Shoes & Accessories
                "filter": "conditions:{USED}",
                "sort": "newlyListed",
                "limit": PAGE_SIZE,
                "offset": offset,
                "fieldgroups": "EXTENDED",
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json()


def _extract_images(item: dict) -> list[str]:
    """Pull all full-size image URLs from an item summary."""
    urls = []
    main = item.get("image", {}).get("imageUrl", "")
    if main:
        urls.append(re.sub(r"s-l\d+", "s-l1600", main))
    for img in item.get("additionalImages", []):
        url = img.get("imageUrl", "")
        if url:
            urls.append(re.sub(r"s-l\d+", "s-l1600", url))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped[:MAX_IMAGES_PER_LISTING]


def _parse_item(item: dict) -> dict:
    label = extract_label(item.get("title", ""))
    return {
        "item_id": item.get("legacyItemId") or item.get("itemId", "").split("|")[1],
        "title": item.get("title", ""),
        "price": f"{item.get('price', {}).get('currency', '')} {item.get('price', {}).get('value', '')}",
        "url": item.get("itemWebUrl", ""),
        "condition": item.get("condition", ""),
        "label": {
            "start": label.start,
            "end": label.end,
            "confidence": label.confidence,
            "raw": label.raw,
        } if label else None,
        "_image_urls": _extract_images(item),
    }


async def scrape_listing(
    client: httpx.AsyncClient,
    listing: dict,
) -> bool:
    item_dir = DATA_DIR / listing["item_id"]
    if (item_dir / "metadata.json").exists():
        return False

    image_urls = listing.pop("_image_urls", [])
    if not image_urls:
        return False

    item_dir.mkdir(parents=True, exist_ok=True)
    images_dir = item_dir / "images"
    images_dir.mkdir(exist_ok=True)

    saved = []
    for idx, url in enumerate(image_urls):
        try:
            r = await client.get(url, timeout=15)
            if r.status_code == 200:
                path = images_dir / f"{idx}.jpg"
                path.write_bytes(r.content)
                saved.append(str(path.relative_to(DATA_DIR)))
        except Exception:
            continue

    if not saved:
        item_dir.rmdir()
        return False

    metadata = {**listing, "images": saved, "scraped_at": time.time()}
    (item_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return True


async def run(queries: list[str], max_pages: int, labeled_only: bool = True):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as http:
        ebay = EbayClient(http)

        for query in queries:
            print(f"\nSearching: {query}")
            all_listings: list[dict] = []

            for page in range(max_pages):
                offset = page * PAGE_SIZE
                try:
                    data = await ebay.search(query, offset=offset)
                except httpx.HTTPStatusError as e:
                    print(f"  API error on page {page+1}: {e}")
                    break

                items = data.get("itemSummaries", [])
                if not items:
                    break

                all_listings.extend(_parse_item(i) for i in items)

                total = data.get("total", 0)
                if offset + PAGE_SIZE >= min(total, max_pages * PAGE_SIZE):
                    break

                await asyncio.sleep(0.3)

            to_scrape = (
                [l for l in all_listings if l["label"] is not None]
                if labeled_only else all_listings
            )
            print(f"  {len(all_listings)} listings found, {len(to_scrape)} with date labels")

            saved = 0
            for listing in tqdm(to_scrape, desc="  Downloading"):
                if await scrape_listing(http, listing):
                    saved += 1
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

            print(f"  Saved {saved} new listings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--pages", type=int, default=MAX_PAGES_PER_QUERY)
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--no-label-filter", action="store_true")
    args = parser.parse_args()

    queries = [args.query] if args.query else SEARCH_QUERIES
    asyncio.run(run(queries, args.pages, labeled_only=not args.no_label_filter))


if __name__ == "__main__":
    main()
