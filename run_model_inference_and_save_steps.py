"""
Step-by-step script for running model inference across the test dataset and
saving the generated outputs to a JSON file. This mirrors the step-driven style
used elsewhere in the project: no functions or classes, just sequential steps
with detailed comments for easy modification.
"""

# step-0: import required libraries and configure paths/devices/hyperparameters
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from transformers.utils import logging
logging.set_verbosity_error()

MODEL_BASE = "E:\models\QwenGuard-v1.2-3B"
TEST_DATASET_PATH = Path("data/test_dataset.json")                # dataset with id/input/output fields
OUTPUT_JSON_PATH = Path("data/model_outputs/7B_model_outputs.json")  # where to save model predictions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 512                                              # generation length
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 50
NUM_BEAMS = 2
SEED = 5388797                                                   # seed for deterministic sampling order


# step-1: load the full test dataset to run inference on every entry
if not TEST_DATASET_PATH.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DATASET_PATH}")
with TEST_DATASET_PATH.open("r", encoding="utf-8") as file:
    test_entries = json.load(file)
if not isinstance(test_entries, list) or not test_entries:
    raise ValueError(f"Test dataset is empty or malformed: {TEST_DATASET_PATH}")
print(f"Loaded {len(test_entries)} test entries from {TEST_DATASET_PATH}.")


# step-2: initialize processor and model, then optionally load fine-tuned weights
processor = AutoProcessor.from_pretrained(MODEL_BASE)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_BASE, torch_dtype="auto", device_map="auto"
)
model.eval()

# step-3: set up generation configuration shared across all samples
common_generation_kwargs = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "num_beams": NUM_BEAMS,
    "use_cache": True,
}

# step-4: iterate over every test sample, run inference, and collect outputs
results = []
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

for sample in test_entries:
    sample_id = sample.get("id", "<unknown>")
    input_block = sample.get("input") or {}
    policy_text = input_block.get("policy")
    image_path = input_block.get("image")

    if not policy_text or not image_path:
        print(f"Skipping sample {sample_id}: missing policy or image path.")
        continue

    image = Image.open(Path(image_path)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": policy_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **common_generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

    decoded_list = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    decoded = decoded_list[0] if decoded_list else ""

    # attempt to normalize JSON-like responses so the evaluation script can parse them
    normalized_output = decoded
    try:
        normalized_output = json.dumps(json.loads(decoded), ensure_ascii=False)
    except Exception:
        pass

    results.append({
        "id": sample_id,
        "output": normalized_output,
    })

    print(f"Finished sample {sample_id} -> output length {len(decoded)} chars")


# step-5: write all model outputs to disk for later comparison with labels
OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} model predictions to {OUTPUT_JSON_PATH}.")
