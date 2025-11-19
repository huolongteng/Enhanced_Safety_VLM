"""Run inference on a random test sample using the fine-tuned student model."""

import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file
from PIL import Image
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers.utils import logging

logging.set_verbosity_error()

# step-0: set runtime variables instead of command-line arguments (for running in IDEs like PyCharm)
MODEL_BASE = "E:/models/llava-onevision-qwen2-0.5b-ov-hf"  # path to the base model checkpoint
STATE_DICT_PATH = "outputs-1141/model.safetensors"          # path to the fine-tuned student weights (.safetensors)
TEST_DIR = "data/test"                                      # folder containing test JPG images
POLICY_PATH = "policy.json"                                 # JSON file with a 'policies' list
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # computation device
SEED = 5388797                                                   # random seed for sampling an image
MAX_NEW_TOKENS = 512
TEST_DATASET_PATH = 'data/test_dataset.json'# generation length

# step-1: load the test dataset entries
TEST_DATASET_FILE = Path(TEST_DATASET_PATH)
if not TEST_DATASET_FILE.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DATASET_FILE}")
with TEST_DATASET_FILE.open("r", encoding="utf-8") as file:
    test_entries = json.load(file)
if not isinstance(test_entries, list) or not test_entries:
    raise ValueError(f"Test dataset is empty or malformed: {TEST_DATASET_FILE}")
print(f"Loaded {len(test_entries)} test entries from {TEST_DATASET_FILE}.")

# step-2: pick a random sample and read its fields (image path, policy, expected output)
random.seed(SEED)
sample = random.choice(test_entries)
sample_id = sample.get("id", "<unknown>")
input_block = sample.get("input") or {}
policy_text = input_block.get("policy")
image_path = input_block.get("image")
expected_output = sample.get("output")
if not policy_text or not image_path or expected_output is None:
    raise ValueError(f"Sample {sample_id} is missing required fields.")
print(f"Selected sample: {sample_id}")
print(f"Image path: {image_path}")

# step-3: load processor and build the chat-style prompt with the sample policy
processor = AutoProcessor.from_pretrained(MODEL_BASE)
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

# step-4: initialize the model from config and load the fine-tuned state dict
config = AutoConfig.from_pretrained(MODEL_BASE)
model = LlavaOnevisionForConditionalGeneration(config)
state_dict = load_file(STATE_DICT_PATH, device="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# step-5: prepare the image and text inputs, then move everything to the target device
device = torch.device(DEVICE)
model.to(device)
image = Image.open(Path(image_path)).convert("RGB")
inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# step-6: run generation with the reference hyperparameters
generation_config = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 2,
    "use_cache": True,
}
with torch.no_grad():
    output = model.generate(**inputs, **generation_config)

# step-7: decode and print the model response, then show the ground truth output
decoded = processor.decode(output[0], skip_special_tokens=True)
print("\nGenerated response:\n-------------------")
print(decoded)
print("\nGround truth output:\n--------------------")
print(expected_output)
