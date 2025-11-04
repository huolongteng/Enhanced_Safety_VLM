from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import json
from train_test_script import count_jpg_in_folder, get_dirs_with_jpg
import random

with open('policy.json', 'r') as f:
    policy = json.load(f)['policies'][0]

policy = "What is shown in the image? Provide a detailed description."
import torch
from transformers import AutoConfig
config = AutoConfig.from_pretrained(r"E:\models\llava-onevision-qwen2-0.5b-ov-hf")
model = LlavaOnevisionForConditionalGeneration(config)
state_dict = torch.load("exp_1104_50_epochs/student_model_state.pt", map_location='cpu')
model.load_state_dict(state_dict, strict=False)

# model = LlavaOnevisionForConditionalGeneration.from_pretrained(r"E:\models\llava-onevision-qwen2-0.5b-ov-hf")
# model = LlavaOnevisionForConditionalGeneration.from_pretrained(r"E:\models\LlavaGuard-v1.2-0.5B-OV-hf")
model.eval()

processor = AutoProcessor.from_pretrained('E:\models\llava-onevision-qwen2-0.5b-ov-hf')

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

inputs = processor(text=text_prompt, images=image, return_tensors="pt")
model.to('cuda:0')
inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
# Generate
hyperparameters = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 2,
    "use_cache": True,
}
output = model.generate(**inputs, **hyperparameters)
print(processor.decode(output[0], skip_special_tokens=True))



