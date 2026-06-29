"""Heuristic parsing of OCR text lines into individual purchased item names.

Korean receipts vary a lot by POS vendor, so this is intentionally simple and
rule-based: keep lines that look like "<name> ... <price>", drop the obvious
header/footer/total lines. It produces the item-name strings that the text
classifier then maps to categories.
"""

from __future__ import annotations

import re

_PRICE = re.compile(r"\d{1,3}(?:[,.]\d{3})+|\d{3,}")
# Lines containing these tokens are receipt metadata, not products.
_STOP_TOKENS = (
    "합계", "총액", "총 액", "소계", "부가세", "과세", "면세", "공급가",
    "결제", "카드", "현금", "승인", "거스름", "받을돈", "할인", "포인트",
    "사업자", "대표", "전화", "주소", "tel", "사업장", "영수증", "교환",
    "감사", "방문", "매장", "점", "no.", "번호", "일시", "고객",
)


def _looks_like_item(line: str) -> bool:
    low = line.lower()
    if any(tok in low for tok in _STOP_TOKENS):
        return False
    if not _PRICE.search(line):
        return False
    # Must contain at least one Hangul or 2+ letters (a real product name).
    if re.search(r"[가-힣]", line):
        return True
    return bool(re.search(r"[A-Za-z]{2,}", line))


def _extract_name(line: str) -> str:
    """Strip trailing quantity/price columns, keep the leading name."""
    m = _PRICE.search(line)
    name = line[: m.start()] if m else line
    name = re.sub(r"\s{2,}", " ", name)
    name = name.strip(" -*xX·.\t")
    return name.strip()


def parse_items(lines: list[str], min_name_len: int = 2) -> list[str]:
    items: list[str] = []
    for line in lines:
        if not _looks_like_item(line):
            continue
        name = _extract_name(line)
        if len(name) >= min_name_len:
            items.append(name)
    return items
