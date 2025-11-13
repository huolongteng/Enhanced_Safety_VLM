"""Utilities for loading the teacher/student models and shared processor."""
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

def load_model_and_processor(teacher_path: str, student_path: str):
    """Load teacher and student checkpoints alongside a shared processor."""
    teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained(teacher_path)
    student_model = LlavaOnevisionForConditionalGeneration.from_pretrained(student_path)
    processor = AutoProcessor.from_pretrained(teacher_path)

    return teacher_model, student_model, processor