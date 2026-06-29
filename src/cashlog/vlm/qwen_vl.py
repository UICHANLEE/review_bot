"""Qwen2.5-VL wrapper.

IMPORTANT: Qwen2.5-VL is a vision-language model, so it must be loaded with
`Qwen2_5_VLForConditionalGeneration` (NOT `AutoModelForCausalLM`, which only
supports text-only causal LMs and raises "Unrecognized configuration class
Qwen2_5_VLConfig").

This model is too heavy for real-time on-device use; here it serves two roles:
  1. server-side fallback when the on-device models are not confident, and
  2. auto-labeling engine to bootstrap a dataset (Phase 2).
"""

from __future__ import annotations

import json
import re

from ..categories import Taxonomy
from ..config import DEFAULT_VLM_LORA, DEFAULT_VLM_MODEL, pick_device


class QwenVL:
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    @classmethod
    def load(
        cls,
        model_name: str = DEFAULT_VLM_MODEL,
        lora_id: str | None = DEFAULT_VLM_LORA,
        device: str | None = None,
        dtype: str = "auto",
    ) -> "QwenVL":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = pick_device(device)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
        if lora_id:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_id)
        model = model.eval().to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return cls(model, processor, device)

    def generate(self, image, prompt: str, max_new_tokens: int = 512) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def transcribe(self, image) -> str:
        """Plain OCR transcription, one item/line per row."""
        return self.generate(
            image,
            "이 영수증의 모든 텍스트를 위에서 아래 순서대로 한 줄씩 그대로 옮겨 적어줘.",
        )

    def label_image(self, image, taxonomy: Taxonomy, input_type: str = "auto") -> dict:
        """Return structured {input_type, items:[{name, category_id}]} for bootstrapping."""
        cats = ", ".join(f"{c.id}({c.name_ko})" for c in taxonomy)
        prompt = (
            "다음 이미지를 분석해줘. 영수증이면 구매 품목을, 상품 사진이면 주요 상품을 찾아줘.\n"
            f"각 상품을 아래 카테고리 중 하나의 id로 분류해줘: {cats}.\n"
            "반드시 아래 JSON 형식으로만 답해줘:\n"
            '{"input_type": "receipt|product", '
            '"items": [{"name": "상품명", "category_id": "카테고리id"}]}'
        )
        raw = self.generate(image, prompt)
        return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    """Best-effort extraction of a JSON object from model output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"input_type": "unknown", "items": [], "_raw": raw}
