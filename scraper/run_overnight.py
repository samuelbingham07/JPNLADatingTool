"""
Overnight scrape run with a wall-clock time cap.

Usage:
    python -m scraper.run_overnight --hours 5
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
    SEARCH_QUERIES, MODERN_QUERIES,
)
from scraper.label_extractor import extract as extract_label
from scraper.ebay_scraper import EbayClient, _parse_item, scrape_listing

PAGE_SIZE = 100


async def run_timed(hours: float, labeled_only: bool = True, queries=SEARCH_QUERIES, pages: int = 10):
    deadline = time.time() + hours * 3600
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    start = time.time()

    async with httpx.AsyncClient(follow_redirects=True) as http:
        ebay = EbayClient(http)

        for query in queries:
            if time.time() >= deadline:
                break

            print(f"\n[{int((time.time()-start)/60)}min] Searching: {query}")
            all_listings = []

            for page in range(pages):
                if time.time() >= deadline:
                    break
                offset = page * PAGE_SIZE
                try:
                    data = await ebay.search(query, offset=offset)
                except Exception as e:
                    print(f"  Error: {e}")
                    break

                items = data.get("itemSummaries", [])
                if not items:
                    break
                all_listings.extend(_parse_item(i) for i in items)

                total = data.get("total", 0)
                if offset + PAGE_SIZE >= total:
                    break
                await asyncio.sleep(0.3)

            to_scrape = [l for l in all_listings if l["label"] is not None] if labeled_only else all_listings
            print(f"  {len(all_listings)} listings, {len(to_scrape)} labeled")

            for listing in tqdm(to_scrape, desc="  Downloading"):
                if time.time() >= deadline:
                    print("\nTime limit reached.")
                    break
                if await scrape_listing(http, listing):
                    total_saved += 1
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

    elapsed = (time.time() - start) / 3600
    print(f"\nDone. {total_saved} new listings saved in {elapsed:.1f}h.")

    # Update memory log
    log = Path(__file__).parent.parent / "data" / "scrape_log.json"
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "saved": total_saved, "hours": round(elapsed, 2)}
    entries = json.loads(log.read_text()) if log.exists() else []
    entries.append(entry)
    log.write_text(json.dumps(entries, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--no-label-filter", action="store_true")
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--set", choices=["default", "modern", "2010s"], default="default")
    args = parser.parse_args()
    print(f"Running for up to {args.hours} hours. Press Ctrl+C to stop early.")
    queries = {"default": SEARCH_QUERIES, "modern": MODERN_QUERIES,
               "2010s": [q for q in MODERN_QUERIES if "201" in q]}[args.set]
    asyncio.run(run_timed(args.hours, labeled_only=not args.no_label_filter, queries=queries, pages=args.pages))


if __name__ == "__main__":
    main()
