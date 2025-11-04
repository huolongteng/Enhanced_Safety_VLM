# This block is for loading models.
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from transformers import AutoConfig
import random
from train_test_script import count_jpg_in_folder, get_dirs_with_jpg
from PIL import Image
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
# layers_to_hook = {
#     "vision_out": "vision_tower",
#     "projector": "multi_modal_projector",
#     "llm_0": "language_model.model.layers.0",
#     "llm_last": "language_model.model.layers.-1",
#     "llm_head": "language_model.lm_head",
# }



# Forward pass through both models.
with torch.no_grad():
    output_teacher = model_teacher.generate(**inputs)
    output_student = model_student.generate(**inputs)

# Compare outputs and intermediate features.
