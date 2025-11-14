from pathlib import Path
from load_models_processors import load_model_and_processor
from training_dataset import *
import torch
import random
from transformers.utils import logging
logging.set_verbosity_error()


TEACHER_MODEL_PATH = "E:/models/LlavaGuard-v1.2-0.5B-OV-hf"
STUDENT_MODEL_PATH = "E:/models/llava-onevision-qwen2-0.5b-ov-hf"
NUM_EPOCHS = 3
BATCH_SIZE = 1
TRAIN_JSON_PATH = "data/train_dataset.json"
TEST_JSON_PATH = "data/test_dataset.json"
LEARNING_RATE = 5e-5
IMAGE_SIZE = 256
OUTPUT_DIR = Path("output")
SEED = 2025
GRADIENT_ACCUMULATION_STEPS = 32
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MIN_EPOCHS = 1
EARLY_STOPPING_RESTORE_BEST = True

_, student_model, processor = load_model_and_processor(
    TEACHER_MODEL_PATH, STUDENT_MODEL_PATH
)

def seed_everything(seed):
    """Seed all common random number generators."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)


train_dataloader = create_dataloader(
    TRAIN_JSON_PATH,
    processor=processor,
    batch_size=BATCH_SIZE,
    padding=False,
    image_size=IMAGE_SIZE,
    add_generation_prompt=False,
)

test_dataloader = create_dataloader(
    TRAIN_JSON_PATH,
    processor=processor,
    batch_size=BATCH_SIZE,
    padding=False,
    image_size=IMAGE_SIZE,
    add_generation_prompt=False,
)


