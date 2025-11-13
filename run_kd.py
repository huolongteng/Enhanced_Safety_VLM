"""Entry point for the supervised fine-tuning training script."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers.utils import logging as hf_logging

from dataset import create_policy_dataloader, load_split_entries
from load_models import load_model_and_processor
from train import (
    DistillationStats,
    EarlyStoppingConfig,
    LoraAdapterSettings,
    apply_lora_adapters,
    run_kd_training,
    save_training_artifacts,
    seed_everything,
)

from transformers.utils import logging

logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("data")
DEFAULT_TRAIN_DATASET_PATH = DEFAULT_DATA_DIR / "train_dataset.json"
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_DIR
DEFAULT_MODEL_PATH = "E:/models/llava-onevision-qwen2-0.5b-ov-hf"
DEFAULT_PROCESSOR_PATH = None
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 7e-4
DEFAULT_STEP_PLOT_STRIDE = 10
DEFAULT_IMAGE_SIZE = 256
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_SEED = 2025
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 32
DEFAULT_ENABLE_LORA = True
DEFAULT_ENABLE_EARLY_STOPPING = True
DEFAULT_EARLY_STOPPING_PATIENCE = 2
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
DEFAULT_EARLY_STOPPING_MIN_EPOCHS = 1
DEFAULT_EARLY_STOPPING_RESTORE_BEST = True


@dataclass(frozen=True)
class KDConfig:
    """Collect configuration values for the supervised run."""

    train_dataset_path: Path = DEFAULT_TRAIN_DATASET_PATH
    image_root: Path = DEFAULT_IMAGE_ROOT
    model_path: str = DEFAULT_MODEL_PATH
    processor_path: str | None = DEFAULT_PROCESSOR_PATH
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    step_plot_stride: int = DEFAULT_STEP_PLOT_STRIDE
    image_size: int = DEFAULT_IMAGE_SIZE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = DEFAULT_SEED
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    enable_lora: bool = DEFAULT_ENABLE_LORA
    enable_early_stopping: bool = DEFAULT_ENABLE_EARLY_STOPPING
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    early_stopping_min_epochs: int = DEFAULT_EARLY_STOPPING_MIN_EPOCHS
    early_stopping_restore_best: bool = DEFAULT_EARLY_STOPPING_RESTORE_BEST


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------


def main(config: KDConfig | None = None) -> DistillationStats:
    cfg = config or KDConfig()

    seed_everything(cfg.seed)
    hf_logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    student_model, processor = load_model_and_processor(
        cfg.model_path,
        processor_path=cfg.processor_path,
    )

    entries = load_split_entries(cfg.train_dataset_path, cfg.image_root)

    missing_images = [
        entry.image_path
        for entry in entries
        if not (cfg.image_root / entry.image_path).exists()
    ]
    if missing_images:
        missing_str = ", ".join(str(path) for path in missing_images[:5])
        raise FileNotFoundError(
            "Some image files referenced in the dataset JSON are missing. "
            f"First missing entries: {missing_str}"
        )

    dataloader = create_policy_dataloader(
        entries,
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

    stats = run_kd_training(
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
