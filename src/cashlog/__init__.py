"""cashlog-auto: product recognition and expense-category recommendation.

Two input tracks (receipt photos and product photos) feed a shared category
taxonomy. The package is organized so that the same high-level pipeline can run
either fully on-device (compact models) or fall back to a server-side VLM.
"""

from .types import CategoryScore, ProductItem, RecognitionResult

__all__ = ["CategoryScore", "ProductItem", "RecognitionResult"]
