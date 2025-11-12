"""Entry point for training the student model with supervised KD signals."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers.utils import logging as hf_logging

from dataset_generate import (
    DATA_DIR,
    create_policy_dataloader,
    load_dataset_entries,
)
from load_models import load_model_and_processor
from train_utils import (
    DistillationStats,
    EarlyStoppingConfig,
    LoraAdapterSettings,
    LossWeights,
    apply_lora_adapters,
    run_kd_training,
    save_training_artifacts,
    seed_everything,
)

from transformers.utils import logging
logging.set_verbosity_error()

DEFAULT_TRAIN_JSON = DATA_DIR / "train_dataset.json"
DEFAULT_EVAL_JSON = DATA_DIR / "test_dataset.json"

# ---------------------------------------------------------------------------
# Inline defaults so the script can be launched directly from an IDE
# ---------------------------------------------------------------------------

TEACHER_MODEL_PATH = "E:\models\LlavaGuard-v1.2-0.5B-OV-hf"
STUDENT_MODEL_PATH = "E:\models\llava-onevision-qwen2-0.5b-ov-hf"
NUM_EPOCHS = 3
BATCH_SIZE = 1
IMAGE_SIZE = 256
LEARNING_RATE = 5e-5
TEMPERATURE = 2.0
PROJECTOR_LOSS_WEIGHT = 0.0
GRADIENT_ACCUMULATION_STEPS = 32
HARD_LOSS_WEIGHT = 0.5
SOFT_LOSS_WEIGHT = 0.5
OUTPUT_DIR = Path("training_outputs")
SEED = 2025
ENABLE_LORA = True
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MIN_EPOCHS = 1
EARLY_STOPPING_RESTORE_BEST = True


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values that control the training workflow."""

    teacher_model_path: str = TEACHER_MODEL_PATH
    student_model_path: str = STUDENT_MODEL_PATH
    train_json: Path = DEFAULT_TRAIN_JSON
    eval_json: Path | None = DEFAULT_EVAL_JSON
    num_epochs: int = 3
    batch_size: int = 1
    image_size: int = 256
    learning_rate: float = 5e-5
    temperature: float = 2.0
    projector_loss_weight: float = 0.0
    gradient_accumulation_steps: int = 32
    hard_loss_weight: float = HARD_LOSS_WEIGHT
    soft_loss_weight: float = SOFT_LOSS_WEIGHT
    output_dir: Path = Path("training_outputs")
    seed: int = 2025
    enable_lora: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 1
    early_stopping_restore_best: bool = True


def _validate_training_defaults() -> None:
    fields = TrainingConfig.__dataclass_fields__

    assert (
        fields["hard_loss_weight"].default == HARD_LOSS_WEIGHT
    ), "TrainingConfig.hard_loss_weight default must stay aligned with CLI defaults"
    assert (
        fields["soft_loss_weight"].default == SOFT_LOSS_WEIGHT
    ), "TrainingConfig.soft_loss_weight default must stay aligned with CLI defaults"


_validate_training_defaults()


def _build_dataloader(
    json_path: Path,
    processor,
    *,
    batch_size: int,
    image_size: int,
    shuffle: bool,
) -> Iterable:
    entries = load_dataset_entries(json_path)
    if not entries:
        raise ValueError(f"Dataset at {json_path} is empty; cannot train.")

    return create_policy_dataloader(
        entries,
        processor,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        include_labels=True,
    )


def _ensure_model_paths(config: TrainingConfig) -> None:
    if "path/to" in config.teacher_model_path:
        raise ValueError(
            "Please update TEACHER_MODEL_PATH or provide --teacher-model to point to your checkpoint."
        )
    if "path/to" in config.student_model_path:
        raise ValueError(
            "Please update STUDENT_MODEL_PATH or provide --student-model to point to your checkpoint."
        )


def main(config: TrainingConfig | None = None) -> DistillationStats:
    """Execute the supervised knowledge distillation routine."""

    cfg = config or TrainingConfig()

    _ensure_model_paths(cfg)

    seed_everything(cfg.seed)
    hf_logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_json = cfg.train_json
    eval_json = cfg.eval_json

    teacher_model, student_model, processor = load_model_and_processor(
        cfg.teacher_model_path,
        cfg.student_model_path,
    )

    train_loader = _build_dataloader(
        train_json,
        processor,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        shuffle=True,
    )
    if hasattr(train_loader, "dataset"):
        try:
            train_size = len(train_loader.dataset)
        except TypeError:
            train_size = "unknown"
        else:
            print(f"Loaded {train_size} training samples from {train_json}")

    if eval_json is not None and Path(eval_json).exists():
        try:
            eval_entries = load_dataset_entries(Path(eval_json))
            print(f"Loaded {len(eval_entries)} evaluation samples from {eval_json}")
        except Exception as exc:  # pragma: no cover - logging only
            print(f"Warning: failed to load evaluation dataset ({exc})")

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
        teacher_model,
        student_model,
        train_loader,
        device=device,
        learning_rate=cfg.learning_rate,
        temperature=cfg.temperature,
        num_epochs=cfg.num_epochs,
        projector_loss_weight=cfg.projector_loss_weight,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        early_stopping=early_stopping_cfg,
        loss_weights=LossWeights(
            hard=cfg.hard_loss_weight,
            soft=cfg.soft_loss_weight,
        ),
    )

    save_training_artifacts(
        student_model,
        stats,
        cfg.output_dir,
        step_stride=max(1, cfg.gradient_accumulation_steps),
    )

    return stats


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the student model with supervised KD")
    parser.add_argument(
        "teacher_model",
        nargs="?",
        default=TEACHER_MODEL_PATH,
        help="Path or identifier of the teacher model",
    )
    parser.add_argument(
        "student_model",
        nargs="?",
        default=STUDENT_MODEL_PATH,
        help="Path or identifier of the student model",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=DEFAULT_TRAIN_JSON,
        help="Path to the training dataset JSON",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=DEFAULT_EVAL_JSON,
        help="Path to the evaluation dataset JSON (optional)",
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Input image resolution")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Optimizer learning rate",
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="KD temperature")
    parser.add_argument(
        "--projector-loss-weight",
        type=float,
        default=PROJECTOR_LOSS_WEIGHT,
        help="Weight applied to the projector alignment loss",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help="Number of batches to accumulate before optimizer step",
    )
    parser.add_argument(
        "--hard-loss-weight",
        type=float,
        default=HARD_LOSS_WEIGHT,
        help="Weight for the supervised cross-entropy loss",
    )
    parser.add_argument(
        "--soft-loss-weight",
        type=float,
        default=SOFT_LOSS_WEIGHT,
        help="Weight for the distillation loss",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where checkpoints and plots will be stored",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA adapters and train the full student model",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable the early stopping heuristic",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Patience (in epochs) for early stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
        help="Minimum improvement required to reset patience",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=EARLY_STOPPING_MIN_EPOCHS,
        help="Minimum epochs before early stopping can trigger",
    )
    parser.add_argument(
        "--no-restore-best",
        action="store_true",
        help="Do not restore the best checkpoint after early stopping",
    )

    args = parser.parse_args(argv)

    return TrainingConfig(
        teacher_model_path=args.teacher_model,
        student_model_path=args.student_model,
        train_json=args.train_json,
        eval_json=args.eval_json,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        projector_loss_weight=args.projector_loss_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        hard_loss_weight=args.hard_loss_weight,
        soft_loss_weight=args.soft_loss_weight,
        output_dir=args.output_dir,
        seed=args.seed,
        enable_lora=not args.no_lora,
        enable_early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_restore_best=not args.no_restore_best,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = parse_args()
    else:
        config = TrainingConfig()
    main(config)

