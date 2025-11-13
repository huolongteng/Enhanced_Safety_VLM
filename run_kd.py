"""Entry point for the supervised fine-tuning training script."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, UnidentifiedImageError
import torch
from transformers.utils import logging as hf_logging

from dataset import create_policy_dataloader, load_split_entries
from load_models import load_student_model_and_processor
from train import (
    EarlyStoppingConfig,
    LoraAdapterSettings,
    apply_lora_adapters,
    run_supervised_training,
    save_training_artifacts,
    seed_everything,
    TrainingStats,
)

from transformers.utils import logging
logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_DATASET_PATH = Path("data/train_dataset.json")
IMAGE_ROOT = Path("/data")
STUDENT_MODEL_PATH = "E:/models/llava-onevision-qwen2-0.5b-ov-hf"
NUM_EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 7e-4
STEP_PLOT_STRIDE = 10
IMAGE_SIZE = 256
OUTPUT_DIR = Path(".")
SEED = 2025
GRADIENT_ACCUMULATION_STEPS = 32
ENABLE_LORA = True
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MIN_EPOCHS = 1
EARLY_STOPPING_RESTORE_BEST = True


@dataclass(frozen=True)
class KDConfig:
    """Collect configuration values for the supervised fine-tuning run."""

    train_dataset_path: Path = TRAIN_DATASET_PATH
    image_root: Path = IMAGE_ROOT
    student_model_path: str = STUDENT_MODEL_PATH
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    step_plot_stride: int = STEP_PLOT_STRIDE
    image_size: int = IMAGE_SIZE
    output_dir: Path = OUTPUT_DIR
    seed: int = SEED
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    enable_lora: bool = ENABLE_LORA
    enable_early_stopping: bool = ENABLE_EARLY_STOPPING
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA
    early_stopping_min_epochs: int = EARLY_STOPPING_MIN_EPOCHS
    early_stopping_restore_best: bool = EARLY_STOPPING_RESTORE_BEST


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------


def _filter_valid_images(paths: Iterable[Path]) -> list[Path]:
    """Remove image paths that cannot be opened by PIL."""

    valid_paths: list[Path] = []
    skipped_paths: list[Path] = []

    for path in paths:
        try:
            with Image.open(path) as img:
                img.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            print(f"Skipping unreadable image: {path} ({exc})")
            skipped_paths.append(path)
            continue

        valid_paths.append(path)

    if skipped_paths:
        print(
            "Filtered out {skipped} invalid images; {kept} remain.".format(
                skipped=len(skipped_paths),
                kept=len(valid_paths),
            )
        )
    else:
        print(f"All {len(valid_paths)} images passed validation.")

    return valid_paths


def _collect_sample_paths(
    samples: Sequence[dict],
    image_root: Path,
) -> list[Path]:
    """Resolve sample image paths relative to ``image_root``."""

    resolved: list[Path] = []
    for sample in samples:
        image_rel = sample.get("image")
        if not image_rel:
            continue
        path = Path(image_rel)
        if not path.is_absolute():
            path = image_root / path
        resolved.append(path)
    return resolved


def main(config: KDConfig | None = None) -> TrainingStats:
    cfg = config or KDConfig()

    seed_everything(cfg.seed)
    hf_logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_split_entries(cfg.train_dataset_path, cfg.image_root)
    if not samples:
        raise ValueError("No training samples were loaded from the dataset split.")

    sample_paths = _collect_sample_paths(samples, cfg.image_root)
    if sample_paths:
        _filter_valid_images(sample_paths[:50])

    student_model, processor = load_student_model_and_processor(cfg.student_model_path)

    dataloader = create_policy_dataloader(
        samples,
        cfg.image_root,
        processor,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
    )

    lora_settings = LoraAdapterSettings(enabled=cfg.enable_lora)
    student_model = apply_lora_adapters(student_model, lora_settings)

    early_stopping_cfg = EarlyStoppingConfig(
        enabled=cfg.enable_early_stopping,
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        min_epochs=cfg.early_stopping_min_epochs,
        restore_best_weights=cfg.early_stopping_restore_best,
    )

    stats = run_supervised_training(
        student_model,
        dataloader,
        device=device,
        learning_rate=cfg.learning_rate,
        num_epochs=cfg.num_epochs,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        early_stopping=early_stopping_cfg,
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
