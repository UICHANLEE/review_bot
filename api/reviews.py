from __future__ import annotations

import re
import sys
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cashlog.reviews import (  # noqa: E402
    ReviewRecord,
    export_reviews_xlsx,
    fetch_appstore_reviews,
    fetch_playstore_reviews,
    resolve_playstore_app_id,
)


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            records, filename = collect_reviews(urlparse(self.path).query)
            with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
                export_reviews_xlsx(records, tmp.name)
                tmp.seek(0)
                payload = tmp.read()

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="reviews.xlsx"; filename*=UTF-8\'\'{quote(filename)}',
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            payload = message.encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def collect_reviews(query: str) -> tuple[list[ReviewRecord], str]:
    params = parse_qs(query)
    app_name = _one(params, "app_name")
    store = _one(params, "store", "both")
    country = _one(params, "country", "kr").lower()
    lang = _one(params, "lang", "ko").lower()
    count = _bounded_int(_one(params, "count", "30"), minimum=1, maximum=100)

    if store not in {"both", "playstore", "appstore"}:
        raise ValueError("store must be one of both, playstore, appstore.")
    if not app_name:
        raise ValueError("app_name is required.")

    records: list[ReviewRecord] = []
    if store in {"both", "playstore"}:
        playstore_app_id = resolve_playstore_app_id(app_name, lang=lang, country=country)
        records.extend(
            fetch_playstore_reviews(
                app_id=playstore_app_id,
                lang=lang,
                country=country,
                target_count=count,
            )
        )

    if store in {"both", "appstore"}:
        records.extend(
            fetch_appstore_reviews(
                app_name=app_name,
                country=country,
                target_count=count,
            )
        )

    filename = f"{_slug(app_name)}_reviews.xlsx"
    return records, filename


def _one(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0].strip()


def _bounded_int(value: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        parsed = minimum
    return max(minimum, min(maximum, parsed))


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value).strip("._-")
    return slug or "app"
