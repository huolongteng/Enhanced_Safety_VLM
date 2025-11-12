# This block is about loading a pre-trained model.
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

def load_model_and_processor(teacher_path, student_path):
    teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained(teacher_path)
    student_model = LlavaOnevisionForConditionalGeneration.from_pretrained(student_path)
    processor = AutoProcessor.from_pretrained(teacher_path)

    # Below is for sliding window eval, which we don't use here.
    # Flash-attn is recommended for better performance. But windows do not support flash-attn. Almost.
    teacher_model.config.sliding_window = False
    student_model.config.sliding_window = False
    return teacher_model, student_model, processor