"""Entry point for the knowledge-distillation training script."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers.utils import logging as hf_logging
from dataset import create_policy_dataloader, gather_image_paths, load_policy_text
from load_models import load_model_and_processor
from train import (
    DistillationStats,
    run_kd_training,
    save_training_artifacts,
    seed_everything,
)

from transformers.utils import logging
logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_FOLDER = "tmp"
MAX_IMAGES = 100
TEACHER_MODEL_PATH = "E:/models/LlavaGuard-v1.2-0.5B-OV-hf"
STUDENT_MODEL_PATH = "E:/models/llava-onevision-qwen2-0.5b-ov-hf"
POLICY_PATH = Path("policy.json")
POLICY_INDEX = 0
NUM_EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 7e-5
DISTILL_TEMPERATURE = 2.0
PROJECTOR_LOSS_WEIGHT = 0
STEP_PLOT_STRIDE = 10
IMAGE_SIZE = 256
OUTPUT_DIR = Path(".")
SEED = 2025


@dataclass(frozen=True)
class KDConfig:
    """Collect configuration values for the KD run."""

    image_folder: str = IMAGE_FOLDER
    max_images: int = MAX_IMAGES
    teacher_model_path: str = TEACHER_MODEL_PATH
    student_model_path: str = STUDENT_MODEL_PATH
    policy_path: Path = POLICY_PATH
    policy_index: int = POLICY_INDEX
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    distill_temperature: float = DISTILL_TEMPERATURE
    projector_loss_weight: float = PROJECTOR_LOSS_WEIGHT
    step_plot_stride: int = STEP_PLOT_STRIDE
    image_size: int = IMAGE_SIZE
    output_dir: Path = OUTPUT_DIR
    seed: int = SEED


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------


def main(config: KDConfig | None = None) -> DistillationStats:
    cfg = config or KDConfig()

    seed_everything(cfg.seed)
    hf_logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_text = load_policy_text(cfg.policy_path, cfg.policy_index)

    image_paths = gather_image_paths(cfg.image_folder, cfg.max_images)
    print(f"Loaded {len(image_paths)} image paths for training.")

    teacher_model, student_model, processor = load_model_and_processor(
        cfg.teacher_model_path,
        cfg.student_model_path,
    )

    dataloader = create_policy_dataloader(
        image_paths,
        policy_text,
        processor,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
    )

    stats = run_kd_training(
        teacher_model,
        student_model,
        dataloader,
        device=device,
        learning_rate=cfg.learning_rate,
        temperature=cfg.distill_temperature,
        num_epochs=cfg.num_epochs,
        projector_loss_weight=cfg.projector_loss_weight,
    )

    save_training_artifacts(
        student_model,
        stats,
        cfg.output_dir,
        step_stride=cfg.step_plot_stride,
    )

    return stats


if __name__ == "__main__":
    main()
