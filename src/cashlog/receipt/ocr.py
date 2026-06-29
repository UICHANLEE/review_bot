"""OCR abstraction for the receipt track.

In the shipping apps, OCR runs on-device (iOS Vision / Android ML Kit). For
development, bootstrapping and server fallback we expose pluggable Python
backends behind a common interface so the rest of the pipeline never depends on
a specific OCR engine.
"""

from __future__ import annotations

from typing import Protocol


class OcrBackend(Protocol):
    """Returns the recognized text lines of a receipt image (top-to-bottom)."""

    def read_lines(self, image) -> list[str]: ...


class EasyOcrBackend:
    """Korean+English OCR via easyocr (`pip install -e .[ocr]`)."""

    def __init__(self, languages: tuple[str, ...] = ("ko", "en"), gpu: bool = False):
        import easyocr  # noqa: F401 - validated at import time

        self._reader = easyocr.Reader(list(languages), gpu=gpu)

    def read_lines(self, image) -> list[str]:
        import numpy as np

        arr = np.array(image) if not isinstance(image, (str, bytes)) else image
        results = self._reader.readtext(arr, detail=1, paragraph=False)
        # Sort by vertical position so lines come out in reading order.
        results.sort(key=lambda r: min(p[1] for p in r[0]))
        return [text for _box, text, _conf in results if text.strip()]


class VlmOcrBackend:
    """Fallback OCR that prompts Qwen2.5-VL to transcribe the receipt.

    Heavier but needs no extra OCR dependency; useful for data bootstrapping.
    """

    def __init__(self, vlm=None):
        from ..vlm.qwen_vl import QwenVL

        self._vlm = vlm or QwenVL.load()

    def read_lines(self, image) -> list[str]:
        text = self._vlm.transcribe(image)
        return [ln.strip() for ln in text.splitlines() if ln.strip()]


def get_ocr_backend(name: str = "auto", **kwargs) -> OcrBackend:
    """Factory. `auto` prefers easyocr and falls back to the VLM transcriber."""
    if name == "easyocr":
        return EasyOcrBackend(**kwargs)
    if name == "vlm":
        return VlmOcrBackend(**kwargs)
    if name == "auto":
        try:
            return EasyOcrBackend(**kwargs)
        except Exception:
            return VlmOcrBackend()
    raise ValueError(f"Unknown OCR backend: {name!r}")
