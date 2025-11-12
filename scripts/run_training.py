"""Entry point for training the student model with supervised KD signals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers.utils import logging as hf_logging

from dataset_generate import (
    DATA_DIR,
    create_policy_dataloader,
    load_dataset_entries,
)
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


DEFAULT_TRAIN_JSON = DATA_DIR / "train_dataset.json"
DEFAULT_EVAL_JSON = DATA_DIR / "test_dataset.json"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values that control the training workflow."""

    teacher_model_path: str
    student_model_path: str
    train_json: Path = DEFAULT_TRAIN_JSON
    eval_json: Path | None = DEFAULT_EVAL_JSON
    num_epochs: int = 3
    batch_size: int = 1
    image_size: int = 256
    learning_rate: float = 5e-5
    temperature: float = 2.0
    projector_loss_weight: float = 0.0
    gradient_accumulation_steps: int = 32
    hard_loss_weight: float = 1.0
    soft_loss_weight: float = 1.0
    output_dir: Path = Path("training_outputs")
    seed: int = 2025
    enable_lora: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 1
    early_stopping_restore_best: bool = True


def load_model_and_processor(teacher_path: str, student_path: str):
    """Load teacher/student models alongside the shared processor."""

    teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained(teacher_path)
    student_model = LlavaOnevisionForConditionalGeneration.from_pretrained(student_path)
    processor = AutoProcessor.from_pretrained(teacher_path)

    teacher_model.config.sliding_window = False
    student_model.config.sliding_window = False

    return teacher_model, student_model, processor


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


def main(config: TrainingConfig) -> DistillationStats:
    """Execute the supervised knowledge distillation routine."""

    seed_everything(config.seed)
    hf_logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_json = config.train_json
    eval_json = config.eval_json

    teacher_model, student_model, processor = load_model_and_processor(
        config.teacher_model_path,
        config.student_model_path,
    )

    train_loader = _build_dataloader(
        train_json,
        processor,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
    )

    if eval_json is not None and Path(eval_json).exists():
        try:
            eval_entries = load_dataset_entries(Path(eval_json))
            print(f"Loaded {len(eval_entries)} evaluation samples from {eval_json}")
        except Exception as exc:  # pragma: no cover - logging only
            print(f"Warning: failed to load evaluation dataset ({exc})")

    lora_settings = LoraAdapterSettings(enabled=config.enable_lora)
    student_model = apply_lora_adapters(student_model, lora_settings)

    early_stopping_cfg = EarlyStoppingConfig(
        enabled=config.enable_early_stopping,
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        min_epochs=config.early_stopping_min_epochs,
        restore_best_weights=config.early_stopping_restore_best,
    )

    stats = run_kd_training(
        teacher_model,
        student_model,
        train_loader,
        device=device,
        learning_rate=config.learning_rate,
        temperature=config.temperature,
        num_epochs=config.num_epochs,
        projector_loss_weight=config.projector_loss_weight,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        early_stopping=early_stopping_cfg,
        loss_weights=LossWeights(
            hard=config.hard_loss_weight,
            soft=config.soft_loss_weight,
        ),
    )

    save_training_artifacts(
        student_model,
        stats,
        config.output_dir,
        step_stride=max(1, config.gradient_accumulation_steps),
    )

    return stats


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the student model with supervised KD")
    parser.add_argument("teacher_model", help="Path or identifier of the teacher model")
    parser.add_argument("student_model", help="Path or identifier of the student model")
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
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--image-size", type=int, default=256, help="Input image resolution")
    parser.add_argument("--learning-rate", type=float, default=7e-4, help="Optimizer learning rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD temperature")
    parser.add_argument(
        "--projector-loss-weight",
        type=float,
        default=0.0,
        help="Weight applied to the projector alignment loss",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Number of batches to accumulate before optimizer step",
    )
    parser.add_argument(
        "--hard-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the supervised cross-entropy loss",
    )
    parser.add_argument(
        "--soft-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the distillation loss",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_outputs"),
        help="Directory where checkpoints and plots will be stored",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
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
        default=2,
        help="Patience (in epochs) for early stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset patience",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=1,
        help="Minimum epochs before early stopping can trigger",
    )
    parser.add_argument(
        "--no-restore-best",
        action="store_true",
        help="Do not restore the best checkpoint after early stopping",
    )

    args = parser.parse_args()

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
    config = parse_args()
    main(config)

