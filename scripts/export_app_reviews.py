"""Crawl Play Store/App Store reviews and export them as an Excel file.

Usage:
    python scripts/export_app_reviews.py adidas --count 500

    python scripts/export_app_reviews.py \
        --playstore-app-id com.adidas.app \
        --appstore-app-name adidas \
        --country kr \
        --lang ko \
        --count 500 \
        --out data/reviews/adidas_reviews.xlsx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cashlog.reviews import (  # noqa: E402
    ReviewRecord,
    export_reviews_xlsx,
    fetch_appstore_reviews,
    fetch_playstore_reviews,
    resolve_playstore_app_id,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("app_name", nargs="?", help="App name to search in both stores, e.g. adidas")
    ap.add_argument("--app-name", dest="app_name_option", help="App name to search in both stores")
    ap.add_argument("--store", choices=["both", "playstore", "appstore"], default="both")
    ap.add_argument("--playstore-app-id", help="Google Play package id, e.g. com.adidas.app")
    ap.add_argument("--appstore-app-name", help="Apple App Store app name, e.g. adidas")
    ap.add_argument("--appstore-app-id", type=int, help="Optional Apple App Store numeric app id")
    ap.add_argument("--country", default="kr", help="Two-letter store country code")
    ap.add_argument("--lang", default="ko", help="Two-letter Play Store language code")
    ap.add_argument("--count", type=int, default=500, help="Review count per store")
    ap.add_argument("--score", type=int, choices=[1, 2, 3, 4, 5], help="Only Play Store score filter")
    ap.add_argument("--out")
    args = ap.parse_args()

    app_name = args.app_name_option or args.app_name
    if not app_name and not args.playstore_app_id and not args.appstore_app_name and not args.appstore_app_id:
        ap.error(
            "Pass an app name, --playstore-app-id, --appstore-app-name, or --appstore-app-id."
        )
    if args.count < 1:
        ap.error("--count must be greater than 0.")

    output = args.out or _default_output_path(app_name or args.appstore_app_name or args.playstore_app_id)
    records: list[ReviewRecord] = []

    playstore_app_id = args.playstore_app_id
    if app_name and args.store in {"both", "playstore"} and not playstore_app_id:
        print(f"Searching Play Store app: {app_name}")
        playstore_app_id = resolve_playstore_app_id(app_name, lang=args.lang, country=args.country)
        print(f"Resolved Play Store app id: {playstore_app_id}")

    appstore_app_name = args.appstore_app_name
    if app_name and args.store in {"both", "appstore"} and not appstore_app_name and not args.appstore_app_id:
        appstore_app_name = app_name

    if playstore_app_id and args.store in {"both", "playstore"}:
        print(f"Fetching Play Store reviews: {playstore_app_id}")
        records.extend(
            fetch_playstore_reviews(
                app_id=playstore_app_id,
                lang=args.lang,
                country=args.country,
                target_count=args.count,
                score=args.score,
            )
        )

    if (appstore_app_name or args.appstore_app_id) and args.store in {"both", "appstore"}:
        appstore_label = appstore_app_name or str(args.appstore_app_id)
        print(f"Fetching App Store reviews: {appstore_label}")
        records.extend(
            fetch_appstore_reviews(
                app_name=appstore_app_name,
                country=args.country,
                target_count=args.count,
                app_id=args.appstore_app_id,
            )
        )

    output = export_reviews_xlsx(records, output)
    print(f"Wrote {len(records)} review(s) to {output}")


def _default_output_path(name: str | None) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name or "app").strip("._-")
    return f"data/reviews/{slug or 'app'}_reviews.xlsx"


if __name__ == "__main__":
    main()
