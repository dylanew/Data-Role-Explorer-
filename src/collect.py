"""Collect data-related job postings from the Adzuna API.

Usage (from the repo root):
    python -m src.collect                      # defaults below
    python -m src.collect --country us --pages 10

What it does:
    1. For each search term, request result pages from Adzuna's job search
       endpoint until we hit the page limit or run out of results.
    2. Tag every posting with the search term that found it and when it was
       collected.
    3. Write everything to a dated snapshot in data/raw/ as JSON Lines
       (one posting per line), plus a small metadata file describing the run.

Design notes:
    - Credentials come from environment variables (ADZUNA_APP_ID,
      ADZUNA_APP_KEY). A local .env file is loaded if present; it is
      git-ignored so keys never reach GitHub.
    - We keep the raw API response for each posting untouched. Choosing
      which fields matter is a cleaning decision (src/clean.py), and keeping
      the raw data means we can change that decision later without
      re-collecting.
    - We do NOT deduplicate here. The same job can appear under several
      search terms (e.g. "data analyst" and "BI analyst"); that overlap is
      itself interesting, and deduplication belongs in the cleaning step.
    - Adzuna's free tier allows roughly 25 requests per minute, so we pause
      between requests and back off when the API says we are going too fast.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

API_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

# Each search term is sent as an exact phrase (Adzuna's `what_phrase`), so
# "data analyst" won't match a posting that only says "data" and "analyst"
# far apart. Abbreviations are written out too, because a posting titled
# "Machine Learning Engineer" won't match the phrase "ML engineer".
SEARCH_TERMS = [
    "data scientist",
    "data analyst",
    "data engineer",
    "machine learning engineer",
    "ml engineer",
    "business intelligence analyst",
    "bi analyst",
]

DEFAULT_COUNTRY = "gb"
DEFAULT_PAGES_PER_TERM = 10
RESULTS_PER_PAGE = 50  # the maximum Adzuna allows per request

SECONDS_BETWEEN_REQUESTS = 2.5  # keeps us under ~25 requests per minute
MAX_RETRIES = 4

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_credentials():
    """Read the Adzuna app id and key from the environment (or a .env file)."""
    load_dotenv()
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise SystemExit(
            "Missing ADZUNA_APP_ID / ADZUNA_APP_KEY. Copy .env.example to .env "
            "and fill in your keys from https://developer.adzuna.com/."
        )
    return app_id, app_key


def fetch_page(session, country, term, page, app_id, app_key):
    """Fetch one page of results for one search term.

    Retries with exponential backoff (2s, 4s, 8s, ...) when Adzuna rate-limits
    us (HTTP 429) or has a temporary server error (5xx). Any other error,
    such as wrong credentials, stops the run straight away.
    """
    url = API_URL.format(country=country, page=page)
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what_phrase": term,
        "results_per_page": RESULTS_PER_PAGE,
        "content-type": "application/json",
    }
    for attempt in range(MAX_RETRIES + 1):
        response = session.get(url, params=params, timeout=30)
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < MAX_RETRIES:
            wait = 2 ** (attempt + 1)
            print(f"  HTTP {response.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()


def collect_term(session, country, term, max_pages, app_id, app_key):
    """Collect up to `max_pages` pages of postings for one search term.

    Stops early when a page comes back empty or we've seen every result
    Adzuna reports for this term.
    """
    postings = []
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for page in range(1, max_pages + 1):  # Adzuna pages start at 1
        data = fetch_page(session, country, term, page, app_id, app_key)
        results = data.get("results", [])
        total = data.get("count", 0)

        for posting in results:
            posting["search_term"] = term
            posting["collected_at"] = collected_at
        postings.extend(results)
        print(f"  page {page}: {len(results)} postings ({len(postings)}/{total})")

        if not results or len(postings) >= total:
            break
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    return postings, total


def save_snapshot(postings, metadata, country, out_dir=RAW_DIR):
    """Write postings to data/raw/adzuna_<country>_<YYYY-MM-DD>.jsonl.

    The date in the filename is the collection date, so each run is a
    separate, reproducible snapshot. A matching .meta.json file records how
    the snapshot was made (terms, page limit, counts).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data_path = out_dir / f"adzuna_{country}_{date}.jsonl"
    meta_path = out_dir / f"adzuna_{country}_{date}.meta.json"

    with open(data_path, "w", encoding="utf-8") as f:
        for posting in postings:
            f.write(json.dumps(posting, ensure_ascii=False) + "\n")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return data_path, meta_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--country", default=DEFAULT_COUNTRY,
                        help="Adzuna country code, e.g. gb, us, de (default: %(default)s)")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES_PER_TERM,
                        help="max pages per search term, 50 postings each (default: %(default)s)")
    args = parser.parse_args()

    app_id, app_key = get_credentials()
    session = requests.Session()

    all_postings = []
    per_term = {}
    for term in SEARCH_TERMS:
        print(f"Searching '{term}' in {args.country}")
        postings, total = collect_term(session, args.country, term,
                                       args.pages, app_id, app_key)
        all_postings.extend(postings)
        per_term[term] = {"collected": len(postings), "available": total}
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    metadata = {
        "source": "Adzuna API",
        "country": args.country,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages_per_term": args.pages,
        "results_per_page": RESULTS_PER_PAGE,
        "search_terms": per_term,
        "total_postings": len(all_postings),
        "unique_ids": len({p["id"] for p in all_postings}),
    }
    data_path, meta_path = save_snapshot(all_postings, metadata, args.country)
    print(f"\nSaved {len(all_postings)} postings "
          f"({metadata['unique_ids']} unique) to {data_path}")


if __name__ == "__main__":
    main()
