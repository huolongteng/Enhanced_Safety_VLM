"""Utilities for loading a fine-tuning model and its processor."""

from __future__ import annotations

from typing import Optional

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

__all__ = ["load_model_and_processor"]


def load_model_and_processor(
    model_path: str,
    *,
    processor_path: Optional[str] = None,
):
    """Load a single model together with the corresponding processor."""

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path)
    processor_source = processor_path or model_path
    processor = AutoProcessor.from_pretrained(processor_source)

    # Ensure the sliding-window configuration is disabled to mirror the
    # behaviour relied upon in the legacy training scripts.
    model.config.sliding_window = False

    return model, processor
