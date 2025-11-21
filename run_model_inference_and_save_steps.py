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
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers.utils import logging

logging.set_verbosity_error()

MODEL_ROLE = "STUDENT"  # toggle between "STUDENT" and "TEACHER" for different checkpoints

if MODEL_ROLE == "STUDENT":
    MODEL_BASE = r"E:\\models\\llava-onevision-qwen2-0.5b-ov-hf"   # base student checkpoint
    STATE_DICT_PATH = "outputs-1141/model.safetensors"               # fine-tuned weights (set to None if unused)
    APPLY_LORA = True                                                 # student uses LoRA adapters
elif MODEL_ROLE == "TEACHER":
    MODEL_BASE = r"E:\\models\\LlavaGuard-v1.2-0.5B-OV-hf"        # teacher checkpoint (no LoRA)
    STATE_DICT_PATH = None                                            # teacher runs as-is
    APPLY_LORA = False
else:
    raise ValueError("MODEL_ROLE must be either 'STUDENT' or 'TEACHER'.")

TEST_DATASET_PATH = Path("data/test_dataset.json")                # dataset with id/input/output fields
OUTPUT_JSON_PATH = Path("data/model_outputs/model_outputs.json")  # where to save model predictions
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
config = AutoConfig.from_pretrained(MODEL_BASE)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(MODEL_BASE, config=config)

if STATE_DICT_PATH and Path(STATE_DICT_PATH).exists():
    print(f"Loading fine-tuned weights from {STATE_DICT_PATH}...")
    if str(STATE_DICT_PATH).endswith(".safetensors"):
        state_dict = load_file(STATE_DICT_PATH)
    else:
        state_dict = torch.load(STATE_DICT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
else:
    print("STATE_DICT_PATH not provided or file missing; using base model weights only.")

model.to(DEVICE)
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

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": policy_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image = Image.open(Path(image_path)).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated = model.generate(**inputs, **common_generation_kwargs)

    decoded = processor.decode(generated[0], skip_special_tokens=True)

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
