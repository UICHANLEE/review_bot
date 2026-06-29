"""Project paths and runtime device selection."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CATEGORIES_PATH = DATA_DIR / "categories.json"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
ARTIFACTS_DIR = ROOT / "artifacts"

# Default backbones. CLIP is English-centric; swap for a multilingual / MobileCLIP
# checkpoint when targeting Korean prompts or on-device export.
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_VLM_LORA = "hoin1218/receipt-qwen25vl-3b-korie-lora"


def pick_device(prefer: str | None = None) -> str:
    """Return the best available torch device string ("mps" on Apple Silicon)."""
    if prefer:
        return prefer
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
