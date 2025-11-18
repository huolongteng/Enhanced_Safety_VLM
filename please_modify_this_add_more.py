from pathlib import Path

import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers.utils import logging

from load_models_processors import load_model_and_processor
from training_dataset import (
    PolicyImageDataset,
    build_policy_collate_fn,
    create_dataloader,
    read_images_policies_responses_paths,
)

logging.set_verbosity_error()


TEACHER_MODEL_PATH = "E:/models/LlavaGuard-v1.2-0.5B-OV-hf"
STUDENT_MODEL_PATH = "E:/models/llava-onevision-qwen2-0.5b-ov-hf"
NUM_EPOCHS = 20
BATCH_SIZE = 1
TRAIN_JSON_PATH = "data/train_dataset.json"
TEST_JSON_PATH = "data/test_dataset.json"
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256
OUTPUT_DIR = Path("outputs-1117")
SEED = 2025
GRADIENT_ACCUMULATION_STEPS = 32
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MIN_EPOCHS = 1
EARLY_STOPPING_RESTORE_BEST = True
VALIDATION_SPLIT_RATIO = 0.15
LOSS_FIGURE_PATH = OUTPUT_DIR / "training_loss_curve.png"
BEST_STATE_PATH = OUTPUT_DIR / "best_model_state.pt"
SOFT_LABEL_WEIGHT = 0.8
HARD_LABEL_WEIGHT = 0.2
KD_TEMPERATURE = 1.0


# step-1: load both teacher/student models and share the processor for distillation
teacher_model, student_model, processor = load_model_and_processor(TEACHER_MODEL_PATH, STUDENT_MODEL_PATH)


# step-2: make training deterministic for reproducibility
def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)


# step-3: prepare output directory for checkpoints and plots
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# step-4: read the full training dataset and create a validation split
train_image_paths, train_policies, train_responses = read_images_policies_responses_paths(TRAIN_JSON_PATH)
full_training_dataset = PolicyImageDataset(train_image_paths, train_policies, train_responses, image_size=IMAGE_SIZE)

if len(full_training_dataset) < 2:
    raise ValueError("Need at least two samples to form train/validation sets.")

validation_size = max(1, int(len(full_training_dataset) * VALIDATION_SPLIT_RATIO))
training_size = len(full_training_dataset) - validation_size

train_subset, val_subset = random_split(
    full_training_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(SEED),
)


# step-5: build the collate function shared across train/val loaders
train_collate_fn = build_policy_collate_fn(
    processor,
    add_generation_prompt=False,
    padding=False,
    return_tensors="pt",
)


# step-6: instantiate the train, validation, and test dataloaders
train_dataloader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=train_collate_fn,
)

validation_dataloader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=train_collate_fn,
)

test_dataloader = create_dataloader(
    TEST_JSON_PATH,
    processor=processor,
    batch_size=BATCH_SIZE,
    padding=False,
    image_size=IMAGE_SIZE,
    add_generation_prompt=False,
)


# step-7: select device and disable caching for training stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(DEVICE)
teacher_model.to(DEVICE)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False


# step-8: freeze every parameter except the student llm block
def freeze_student_vision_and_projector(student_model):
    frozen_components = []
    candidate_attrs = (
        "vision_tower",
        "vision_model",
        "multi_modal_projector",
        "multimodal_projector",
        "vision_projector",
        "mm_projector",
        "projector",
    )
    candidate_parents = [student_model]
    if hasattr(student_model, "model") and getattr(student_model, "model") is not None:
        candidate_parents.append(getattr(student_model, "model"))
    for parent in candidate_parents:
        for attr in candidate_attrs:
            if not hasattr(parent, attr):
                continue
            module = getattr(parent, attr)
            if module is None:
                continue
            parameters = getattr(module, "parameters", None)
            if parameters is None:
                continue
            for param in parameters():
                param.requires_grad = False
            frozen_components.append(attr)
    return frozen_components



frozen_visual_components = freeze_student_vision_and_projector(student_model)



# step-9: configure optimizer only for trainable parameters
TRAINABLE_PARAMETERS = [p for p in student_model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(TRAINABLE_PARAMETERS, lr=LEARNING_RATE)


# step-10: containers for tracking metrics and early stopping
GLOBAL_STEP = 0
TRAINING_STEP_LOSSES = []
EPOCH_VALIDATION_LOSSES = []
EPOCH_TEST_LOSSES = []
BEST_VALIDATION_LOSS = float("inf")
EARLY_STOPPING_COUNTER = 0


def move_batch_to_device(batch, device):
    updated = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            updated[key] = value.to(device)
        else:
            updated[key] = value
    return updated


def compute_kd_loss(student_logits, teacher_logits, attention_mask, temperature=1.0):
    """Compute token-level KL divergence distillation loss with optional masking."""

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_log_probs = student_log_probs.view(-1, student_log_probs.size(-1))
    teacher_probs = teacher_probs.view(-1, teacher_probs.size(-1))

    if attention_mask is not None:
        flat_mask = attention_mask.view(-1).bool()
        if flat_mask.any():
            student_log_probs = student_log_probs[flat_mask]
            teacher_probs = teacher_probs[flat_mask]
        else:
            return torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    return kd * (temperature**2)


def evaluate_dataloader(dataloader):
    student_model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, DEVICE)
            outputs = student_model(**batch)
            total_loss += float(outputs.loss.detach().cpu())
            total_batches += 1
    student_model.train()
    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


# step-11: run the supervised fine-tuning loop with gradient accumulation and early stopping
for epoch_index in range(NUM_EPOCHS):
    student_model.train()
    accumulated_loss = 0.0
    micro_step_counter = 0
    for batch in train_dataloader:
        batch = move_batch_to_device(batch, DEVICE)
        # Run teacher forward pass only for logits to build soft labels without gradients
        with torch.no_grad():
            teacher_inputs = {k: v for k, v in batch.items() if k != "labels"}
            teacher_outputs = teacher_model(
                **teacher_inputs,
                output_hidden_states=False,
                use_cache=False,
            )
            teacher_logits = teacher_outputs.logits.detach()

        # Student computes hard-label cross-entropy and soft-label KL divergence
        student_outputs = student_model(
            **batch,
            output_hidden_states=False,
            use_cache=False,
        )
        student_logits = student_outputs.logits
        hard_loss = student_outputs.loss

        # Align logits and masks for causal LM (predict token t+1) and compute distillation
        shifted_student_logits = student_logits[:, :-1, :]
        shifted_teacher_logits = teacher_logits[:, :-1, :]
        attention_mask = batch.get("attention_mask")
        shifted_attention = attention_mask[:, 1:] if attention_mask is not None else None
        label_mask = (batch["labels"][:, 1:] != -100)
        combined_mask = label_mask if shifted_attention is None else (shifted_attention.bool() & label_mask)
        soft_loss = compute_kd_loss(
            shifted_student_logits,
            shifted_teacher_logits,
            combined_mask,
            temperature=KD_TEMPERATURE,
        )

        loss = SOFT_LABEL_WEIGHT * soft_loss + HARD_LABEL_WEIGHT * hard_loss
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulated_loss += float(loss.detach().cpu())
        micro_step_counter += 1

        if micro_step_counter == GRADIENT_ACCUMULATION_STEPS:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            GLOBAL_STEP += 1
            average_loss = accumulated_loss / micro_step_counter
            TRAINING_STEP_LOSSES.append((GLOBAL_STEP, average_loss))
            accumulated_loss = 0.0
            micro_step_counter = 0

    if micro_step_counter > 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        GLOBAL_STEP += 1
        average_loss = accumulated_loss / micro_step_counter
        TRAINING_STEP_LOSSES.append((GLOBAL_STEP, average_loss))

    validation_loss = evaluate_dataloader(validation_dataloader)
    test_loss = evaluate_dataloader(test_dataloader)
    EPOCH_VALIDATION_LOSSES.append((epoch_index + 1, validation_loss))
    EPOCH_TEST_LOSSES.append((epoch_index + 1, test_loss))
    print(f"epoch={epoch_index + 1} val_loss={validation_loss:.4f} test_loss={test_loss:.4f}")

    if validation_loss + EARLY_STOPPING_MIN_DELTA < BEST_VALIDATION_LOSS:
        BEST_VALIDATION_LOSS = validation_loss
        EARLY_STOPPING_COUNTER = 0
        if EARLY_STOPPING_RESTORE_BEST:
            torch.save(student_model.state_dict(), BEST_STATE_PATH)
    else:
        EARLY_STOPPING_COUNTER += 1

    if ENABLE_EARLY_STOPPING and epoch_index + 1 >= EARLY_STOPPING_MIN_EPOCHS:
        if EARLY_STOPPING_COUNTER > EARLY_STOPPING_PATIENCE:
            break


# step-12: restore the best validation checkpoint if requested
if EARLY_STOPPING_RESTORE_BEST and BEST_STATE_PATH.exists():
    student_model.load_state_dict(torch.load(BEST_STATE_PATH, map_location=DEVICE))


# step-13: persist the fine-tuned student model to the configured output directory
student_model.save_pretrained(OUTPUT_DIR)


# step-14: draw the loss curve sampled every five optimizer steps
if TRAINING_STEP_LOSSES:
    recorded_steps = [item[0] for item in TRAINING_STEP_LOSSES]
    recorded_losses = [item[1] for item in TRAINING_STEP_LOSSES]
    sampled_steps = recorded_steps[::15] if len(recorded_steps) > 1 else recorded_steps
    sampled_losses = recorded_losses[::15] if len(recorded_losses) > 1 else recorded_losses
    if sampled_steps and sampled_steps[-1] != recorded_steps[-1]:
        sampled_steps.append(recorded_steps[-1])
        sampled_losses.append(recorded_losses[-1])
    plt.figure()
    plt.plot(sampled_steps, sampled_losses, marker="o")
    plt.xlabel("Optimizer Step")
    plt.ylabel("Training Loss")
    plt.title("Student LLM Fine-Tuning Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_FIGURE_PATH)
    plt.close()


