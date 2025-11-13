"""Utilities for loading models and their associated processors."""

from __future__ import annotations

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

__all__ = [
    "load_model_and_processor",
    "load_student_model_and_processor",
]


def load_student_model_and_processor(
    student_path: str,
    *,
    processor_path: str | None = None,
):
    """Load a single student checkpoint along with its processor."""

    processor_source = processor_path or student_path

    student_model = LlavaOnevisionForConditionalGeneration.from_pretrained(student_path)
    processor = AutoProcessor.from_pretrained(processor_source)

    student_model.config.sliding_window = False

    return student_model, processor


def load_model_and_processor(teacher_path: str, student_path: str):
    """Load teacher and student checkpoints for compatibility with legacy flows."""

    teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained(teacher_path)
    student_model = LlavaOnevisionForConditionalGeneration.from_pretrained(student_path)
    processor = AutoProcessor.from_pretrained(teacher_path)

    teacher_model.config.sliding_window = False
    student_model.config.sliding_window = False

    return teacher_model, student_model, processor
