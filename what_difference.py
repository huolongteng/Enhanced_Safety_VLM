# This block is for loading models.
import difflib
import random
from typing import Any, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration

from train_test_script import count_jpg_in_folder, get_dirs_with_jpg
# Student model loading.
config = AutoConfig.from_pretrained(r"E:\models\llava-onevision-qwen2-0.5b-ov-hf")
model_student = LlavaOnevisionForConditionalGeneration(config)
state_dict = torch.load("exp_1104_50_epochs/student_model_state.pt", map_location='cpu')
model_student.load_state_dict(state_dict, strict=False)
# Teacher model loading.
model_teacher = LlavaOnevisionForConditionalGeneration.from_pretrained("E:\models\LlavaGuard-v1.2-0.5B-OV-hf")
processor = AutoProcessor.from_pretrained('E:\models\llava-onevision-qwen2-0.5b-ov-hf')
model_teacher.eval().cuda()
model_student.eval().cuda()
# Inputs preparation.
import json
with open('policy.json', 'r') as f:
    policy = json.load(f)['policies'][0]
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": policy},
            ],
    },
]
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
folder_name = "tmp"
count_jpg_in_folder(folder_name)
image_paths = get_dirs_with_jpg(folder_name)
random.seed(5678)
image_paths = random.sample(image_paths, min(1, len(image_paths)))
image = Image.open(image_paths[0]).convert('RGB')
inputs = processor(text=text_prompt, images=image, return_tensors="pt").to('cuda')

# Register hooks to capture intermediate features.


def build_layers_to_hook(model):
    last_layer_idx = len(model.language_model.model.layers) - 1
    return {
        "vision_tower": "vision_tower",
        "multi_modal_projector": "multi_modal_projector",
        "llm_layer_0": "language_model.model.layers.0",
        "llm_layer_last": f"language_model.model.layers.{last_layer_idx}",
        "lm_head": "language_model.lm_head",
    }


layers_to_hook = build_layers_to_hook(model_teacher)


def register_hooks(model, storage_dict, layers_dict):
    hooks = []

    for layer_key, module_name in layers_dict.items():
        module = model.get_submodule(module_name)

        def hook_fn(_, __, output, key=layer_key):
            def detach_output(tensor_like):
                if isinstance(tensor_like, torch.Tensor):
                    return tensor_like.detach()
                if isinstance(tensor_like, (list, tuple)):
                    return type(tensor_like)(detach_output(item) for item in tensor_like)
                if isinstance(tensor_like, dict):
                    return {k: detach_output(v) for k, v in tensor_like.items()}
                return tensor_like

            storage_dict[key] = detach_output(output)

        hooks.append(module.register_forward_hook(hook_fn))

    return hooks


features_teacher: Dict[str, Any] = {}
features_student: Dict[str, Any] = {}
hooks_teacher = register_hooks(model_teacher, features_teacher, layers_to_hook)
hooks_student = register_hooks(model_student, features_student, layers_to_hook)


try:
    # Forward pass through both models.
    with torch.no_grad():
        output_teacher = model_teacher.generate(**inputs)
        output_student = model_student.generate(**inputs)
finally:
    for hook in hooks_teacher + hooks_student:
        hook.remove()

def decode_sequences(token_processor, sequences):
    return token_processor.batch_decode(sequences, skip_special_tokens=True)


def diff_texts(decoded_teacher, decoded_student):
    diffs = []
    max_len = max(len(decoded_teacher), len(decoded_student))
    for idx in range(max_len):
        teacher_text = decoded_teacher[idx] if idx < len(decoded_teacher) else "<missing>"
        student_text = decoded_student[idx] if idx < len(decoded_student) else "<missing>"
        diff_lines = list(
            difflib.unified_diff(
                teacher_text.splitlines(),
                student_text.splitlines(),
                fromfile="teacher",
                tofile="student",
                lineterm="",
            )
        )
        diffs.append(
            {
                "index": idx,
                "teacher": teacher_text,
                "student": student_text,
                "diff": "\n".join(diff_lines) if diff_lines else "No textual differences detected.",
            }
        )
    return diffs


def _flatten_features(features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    flat: Dict[str, torch.Tensor] = {}

    def recurse(item: Any, prefix: str) -> None:
        if item is None:
            return
        if isinstance(item, torch.Tensor):
            flat[prefix] = item
            return
        if isinstance(item, (list, tuple)):
            for idx, value in enumerate(item):
                recurse(value, f"{prefix}[{idx}]")
            return
        if isinstance(item, dict):
            for key, value in item.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                recurse(value, next_prefix)
            return

    for key, value in features.items():
        recurse(value, key)

    return flat


def compare_feature_sets(features_teacher: Dict[str, Any], features_student: Dict[str, Any]):
    teacher_flat = _flatten_features(features_teacher)
    student_flat = _flatten_features(features_student)

    comparison: Dict[str, Dict[str, Any]] = {}

    overlapping_keys = set(teacher_flat.keys()) & set(student_flat.keys())
    for key in sorted(overlapping_keys):
        teacher_tensor = teacher_flat[key]
        student_tensor = student_flat[key]
        if teacher_tensor.shape != student_tensor.shape:
            comparison[key] = {
                "shape_teacher": list(teacher_tensor.shape),
                "shape_student": list(student_tensor.shape),
                "note": "Shape mismatch, metrics not computed.",
            }
            continue

        mse = F.mse_loss(teacher_tensor, student_tensor).item()
        teacher_flattened = teacher_tensor.reshape(1, -1)
        student_flattened = student_tensor.reshape(1, -1)
        cosine = F.cosine_similarity(teacher_flattened, student_flattened).item()
        comparison[key] = {
            "mse": mse,
            "cosine_similarity": cosine,
        }

    for key in sorted(set(teacher_flat.keys()) - overlapping_keys):
        comparison[key] = {"note": "Feature missing from student model."}

    for key in sorted(set(student_flat.keys()) - overlapping_keys):
        comparison[key] = {"note": "Feature missing from teacher model."}

    return comparison


def build_report(text_differences, feature_comparison):
    lines = ["=== Text Output Differences ==="]
    for sample in text_differences:
        lines.append(f"Sample {sample['index']}")
        lines.append("Teacher:")
        lines.append(sample["teacher"])
        lines.append("Student:")
        lines.append(sample["student"])
        lines.append("Diff:")
        lines.append(sample["diff"])
        lines.append("")

    lines.append("=== Feature Differences ===")
    for key in sorted(feature_comparison.keys()):
        lines.append(key)
        metrics = feature_comparison[key]
        for metric_name, metric_value in metrics.items():
            lines.append(f"  {metric_name}: {metric_value}")
        lines.append("")

    return "\n".join(lines)


decoded_teacher = decode_sequences(processor, output_teacher)
decoded_student = decode_sequences(processor, output_student)
text_differences = diff_texts(decoded_teacher, decoded_student)
feature_comparison = compare_feature_sets(features_teacher, features_student)
report = build_report(text_differences, feature_comparison)

print(report)
