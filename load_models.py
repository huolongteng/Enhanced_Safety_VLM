"""Utilities for loading the teacher/student models and shared processor."""

from __future__ import annotations

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

__all__ = ["load_model_and_processor"]


def load_model_and_processor(teacher_path: str, student_path: str):
    """Load teacher and student checkpoints alongside a shared processor."""

    teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained(teacher_path)
    student_model = LlavaOnevisionForConditionalGeneration.from_pretrained(student_path)
    processor = AutoProcessor.from_pretrained(teacher_path)

    # Ensure the sliding-window configuration is disabled to mirror the
    # behaviour relied upon in the legacy training scripts.
    teacher_model.config.sliding_window = False
    student_model.config.sliding_window = False

    return teacher_model, student_model, processor
