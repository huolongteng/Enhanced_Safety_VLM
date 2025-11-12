"""Training helpers for multimodal knowledge distillation with supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup


@dataclass(frozen=True)
class LoraAdapterSettings:
    """Configuration wrapper for optional LoRA fine-tuning."""

    enabled: bool = False
    r: int = 16
    alpha: int = 16
    dropout: float = 0.03
    bias: str = "none"
    target_modules: tuple[str, ...] | None = None


DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class DistillationStats:
    """Container for metrics gathered during training."""

    epoch_losses: list[float]
    step_losses: list[float]
    best_epoch: int | None = None
    best_loss: float | None = None
    early_stopped: bool = False


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Parameters that control the optional early-stopping heuristic."""

    enabled: bool = False
    patience: int = 2
    min_delta: float = 0.0
    min_epochs: int = 1
    restore_best_weights: bool = True


@dataclass(frozen=True)
class LossWeights:
    """Weights applied to the hard (label) and soft (KD) losses."""

    hard: float = 1.0
    soft: float = 1.0


def seed_everything(seed: int) -> None:
    """Seed all common random number generators."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA specific branch
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    import random

    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _move_batch_to_device(batch, device):
    """Move every tensor in batch to ``device`` while keeping other items."""

    if device is None:
        return batch

    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def apply_lora_adapters(student_model, settings: LoraAdapterSettings | None):
    """Wrap ``student_model`` with LoRA adapters when ``settings`` enable it."""

    if settings is None or not settings.enabled:
        return student_model

    try:  # pragma: no cover - optional dependency
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - guarded import
        raise ImportError(
            "LoRA fine-tuning requested but the `peft` package is not installed. "
            "Install it with `pip install peft` to continue."
        ) from exc

    target_modules = settings.target_modules or DEFAULT_LORA_TARGET_MODULES

    lora_config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(student_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def freeze_student_vision_and_projector(student_model) -> list[str]:
    """Freeze the vision tower and projector weights of ``student_model``."""

    frozen_components: list[str] = []

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


def forward_teacher_student(teacher_model, student_model, batch, device):
    """Run a forward pass for both teacher and student on ``batch``."""

    model_inputs = _move_batch_to_device(batch, device)

    teacher_inputs = dict(model_inputs)
    teacher_inputs.pop("labels", None)

    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_inputs)

    student_outputs = student_model(**model_inputs)
    labels = model_inputs.get("labels")

    return teacher_outputs, student_outputs, teacher_inputs, labels


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    labels: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute KL-divergence loss for knowledge distillation."""

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_log_probs = student_log_probs[:, :-1, :].contiguous()
    teacher_probs = teacher_probs[:, :-1, :].contiguous()

    mask = None
    if labels is not None:
        mask = labels[..., 1:] != -100
    elif attention_mask is not None:
        mask = attention_mask[..., 1:].bool()

    student_flat = student_log_probs.view(-1, student_log_probs.size(-1))
    teacher_flat = teacher_probs.view(-1, teacher_probs.size(-1))

    if mask is not None:
        mask_flat = mask.reshape(-1)
        if mask_flat.any():
            student_flat = student_flat[mask_flat]
            teacher_flat = teacher_flat[mask_flat]
        else:
            return torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

    kd = F.kl_div(student_flat, teacher_flat, reduction="batchmean")
    return kd * (temperature**2)


def _extract_projector_hidden_states(model_outputs) -> torch.Tensor | None:
    """Return the projector hidden states tensor if the model exposes it."""

    if model_outputs is None:
        return None

    candidate = None
    for attr in (
        "projector_hidden_states",
        "multi_modal_projector_hidden_states",
        "multimodal_projector_hidden_states",
        "vision_projector_hidden_states",
    ):
        if hasattr(model_outputs, attr):
            candidate = getattr(model_outputs, attr)
            break

    if candidate is None and isinstance(model_outputs, dict):
        for key in (
            "projector_hidden_states",
            "multi_modal_projector_hidden_states",
            "multimodal_projector_hidden_states",
            "vision_projector_hidden_states",
        ):
            if key in model_outputs:
                candidate = model_outputs[key]
                break

    if candidate is None:
        return None

    if isinstance(candidate, (list, tuple)) and candidate:
        candidate = candidate[-1]

    if not isinstance(candidate, torch.Tensor):
        return None

    return candidate


def _reshape_projector_tokens(states: torch.Tensor) -> torch.Tensor:
    """Ensure projector states are shaped as ``(batch, tokens, dim)``."""

    if states.dim() == 3:
        return states
    if states.dim() < 3:
        return states.unsqueeze(1)
    return states.reshape(states.size(0), -1, states.size(-1))


def compute_projector_alignment_loss(
    student_outputs,
    teacher_outputs,
) -> torch.Tensor | None:
    """Cosine + MSE alignment between teacher and student projector tokens."""

    student_states = _extract_projector_hidden_states(student_outputs)
    teacher_states = _extract_projector_hidden_states(teacher_outputs)

    if student_states is None or teacher_states is None:
        return None

    if student_states.dim() < 2 or teacher_states.dim() < 2:
        return None

    min_tokens = min(student_states.size(1), teacher_states.size(1))
    if min_tokens <= 0:
        return None

    student_states = student_states[:, :min_tokens]
    teacher_states = teacher_states[:, :min_tokens]

    student_states = _reshape_projector_tokens(student_states)
    teacher_states = _reshape_projector_tokens(teacher_states)

    teacher_states = teacher_states.detach().to(student_states.dtype)

    student_norm = F.normalize(student_states, dim=-1)
    teacher_norm = F.normalize(teacher_states, dim=-1)

    cosine_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
    loss_cos = 1.0 - cosine_sim
    loss_mse = F.mse_loss(student_states, teacher_states)

    return loss_cos + 0.1 * loss_mse


def distillation_step(
    teacher_model,
    student_model,
    optimizer,
    batch,
    *,
    device,
    temperature: float,
    projector_loss_weight: float,
    loss_weights: LossWeights,
    loss_scale: float = 1.0,
    zero_grad: bool = True,
    step_optimizer: bool = True,
    zero_after_step: bool = True,
) -> float:
    """Perform a single optimization step of supervised knowledge distillation."""

    teacher_outputs, student_outputs, model_inputs, labels = forward_teacher_student(
        teacher_model,
        student_model,
        batch,
        device,
    )

    attention_mask = model_inputs.get("attention_mask")
    kd_loss = compute_kd_loss(
        student_outputs.logits,
        teacher_outputs.logits,
        attention_mask,
        labels=labels,
        temperature=temperature,
    )

    hard_loss = None
    if labels is not None and hasattr(student_outputs, "loss") and student_outputs.loss is not None:
        hard_loss = student_outputs.loss

    projector_loss = None
    distill_loss = kd_loss
    if projector_loss_weight > 0.0:
        projector_loss = compute_projector_alignment_loss(
            student_outputs,
            teacher_outputs,
        )
        if projector_loss is not None:
            mix = max(0.0, min(1.0, projector_loss_weight))
            distill_loss = (1.0 - mix) * kd_loss + mix * projector_loss

    total_loss: torch.Tensor | None = None

    if loss_weights.soft > 0.0:
        kd_component = distill_loss * loss_weights.soft
        total_loss = kd_component if total_loss is None else total_loss + kd_component

    if hard_loss is not None and loss_weights.hard > 0.0:
        hard_component = hard_loss * loss_weights.hard
        total_loss = hard_component if total_loss is None else total_loss + hard_component

    if total_loss is None:
        raise ValueError("Loss weights resulted in zero total loss; nothing to optimize.")

    if zero_grad:
        optimizer.zero_grad()

    scaled_loss = total_loss * loss_scale
    scaled_loss.backward()

    if step_optimizer:
        optimizer.step()
        if zero_after_step:
            optimizer.zero_grad()

    return float(total_loss.detach().item())


def run_kd_training(
    teacher_model,
    student_model,
    dataloader: Iterable,
    *,
    device,
    learning_rate: float,
    temperature: float,
    num_epochs: int,
    projector_loss_weight: float = 1.0,
    gradient_accumulation_steps: int = 1,
    early_stopping: EarlyStoppingConfig | None = None,
    loss_weights: LossWeights | None = None,
) -> DistillationStats:
    """Train ``student_model`` to mimic ``teacher_model`` using ``dataloader``."""

    teacher_model.eval()
    teacher_model.to(device)
    student_model.to(device)

    frozen_parts = freeze_student_vision_and_projector(student_model)
    if frozen_parts:
        print(
            "Frozen student components (vision + projector): "
            + ", ".join(sorted(set(frozen_parts)))
        )

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    scheduler = None

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    global_step = 0

    batch_size = getattr(dataloader, "batch_size", None)

    steps_per_epoch = None
    try:
        steps_per_epoch = len(dataloader)
    except TypeError:  # pragma: no cover - iterables without __len__
        steps_per_epoch = None

    total_batches = None
    total_optimizer_steps = None
    if steps_per_epoch is not None and steps_per_epoch > 0:
        total_batches = steps_per_epoch * num_epochs
        updates_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation_steps)
        total_optimizer_steps = updates_per_epoch * num_epochs
        warmup_steps = max(1, int(0.1 * total_optimizer_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

    progress_bar = tqdm(
        total=total_batches,
        desc="Training",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    early_cfg = early_stopping if early_stopping and early_stopping.enabled else None
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    epochs_without_improve = 0
    stopped_early = False

    weights = loss_weights or LossWeights()

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        num_batches = 0

        progress_bar.set_description(f"Training (epoch {epoch + 1}/{num_epochs})")

        accumulated_batches = 0
        accumulation_loss = 0.0
        current_cycle_size = gradient_accumulation_steps

        for batch_idx, batch in enumerate(dataloader):
            if accumulated_batches == 0:
                if steps_per_epoch is not None and steps_per_epoch > 0:
                    remaining_batches = steps_per_epoch - batch_idx
                    current_cycle_size = min(
                        gradient_accumulation_steps,
                        max(1, remaining_batches),
                    )
                else:
                    current_cycle_size = gradient_accumulation_steps

            accumulated_batches += 1

            zero_grad = accumulated_batches == 1
            should_step = accumulated_batches >= max(1, current_cycle_size)

            loss_scale = 1.0 / float(max(1, current_cycle_size))

            step_loss = distillation_step(
                teacher_model,
                student_model,
                optimizer,
                batch,
                device=device,
                temperature=temperature,
                projector_loss_weight=projector_loss_weight,
                loss_weights=weights,
                loss_scale=loss_scale,
                zero_grad=zero_grad,
                step_optimizer=should_step,
                zero_after_step=should_step,
            )

            running_loss += step_loss
            num_batches += 1
            accumulation_loss += step_loss

            progress_bar.update(1)

            if should_step:
                cycle_batch_count = accumulated_batches
                average_cycle_loss = accumulation_loss / max(1, cycle_batch_count)
                step_losses.append(average_cycle_loss)
                global_step += 1
                accumulation_loss = 0.0
                accumulated_batches = 0

                if scheduler is not None:
                    scheduler.step()

        if accumulated_batches > 0:
            correction = (
                current_cycle_size / max(1, accumulated_batches)
                if current_cycle_size != accumulated_batches
                else 1.0
            )
            if correction != 1.0:
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.mul_(correction)

            optimizer.step()
            optimizer.zero_grad()

            cycle_batch_count = accumulated_batches
            average_cycle_loss = accumulation_loss / max(1, cycle_batch_count)
            step_losses.append(average_cycle_loss)
            global_step += 1
            accumulated_batches = 0
            accumulation_loss = 0.0

            if scheduler is not None:
                scheduler.step()

        epoch_loss = running_loss / max(1, num_batches)
        epoch_losses.append(epoch_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        metrics = [
            f"Epoch {epoch + 1}/{num_epochs}",
            f"average_loss={epoch_loss:.4f}",
            f"lr={current_lr:.6g}",
            f"temperature={temperature:.2f}",
        ]
        if batch_size is not None:
            metrics.append(f"batch_size={batch_size}")
        metrics.append(f"steps={global_step}")
        progress_bar.write(" | ".join(metrics))
        if total_batches is None:
            progress_bar.refresh()

        if early_cfg is not None:
            improved = epoch_loss + early_cfg.min_delta < best_loss
            if improved:
                best_loss = epoch_loss
                best_epoch = epoch
                epochs_without_improve = 0
                if early_cfg.restore_best_weights:
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()
                    }
            else:
                epochs_without_improve += 1

            if (
                (epoch + 1) >= max(1, early_cfg.min_epochs)
                and epochs_without_improve >= max(1, early_cfg.patience)
            ):
                progress_bar.write(
                    "Early stopping triggered: no improvement for "
                    f"{epochs_without_improve} epoch(s). Best loss "
                    f"{best_loss:.4f} observed at epoch {best_epoch + 1 if best_epoch is not None else 'N/A'}."
                )
                stopped_early = True
                break

    progress_bar.close()

    if early_cfg is not None and early_cfg.restore_best_weights and best_state is not None:
        student_model.load_state_dict(best_state)

    return DistillationStats(
        epoch_losses=epoch_losses,
        step_losses=step_losses,
        best_epoch=None if best_epoch is None else best_epoch + 1,
        best_loss=None if best_loss == float("inf") else best_loss,
        early_stopped=stopped_early,
    )


def _save_epoch_plot(epoch_losses: Sequence[float], output_dir: Path) -> None:
    if not epoch_losses:
        return

    epochs = range(1, len(epoch_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_losses, marker="o", label="Epoch Avg Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Knowledge Distillation Training Loss (Epoch)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "loss_curve_epoch.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved epoch loss curve to {plot_path.resolve()}")


def _save_step_plot(step_losses: Sequence[float], output_dir: Path, stride: int) -> None:
    if not step_losses:
        return

    stride = max(1, stride)
    steps = range(1, len(step_losses) + 1)
    steps_plot = steps[::stride]
    losses_plot = step_losses[::stride]
    plt.figure(figsize=(8, 5))
    plt.plot(steps_plot, losses_plot, label=f"Step Loss (every {stride})")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Knowledge Distillation Training Loss (Step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "loss_curve_step.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved step loss curve to {plot_path.resolve()}")


def save_training_artifacts(
    student_model,
    stats: DistillationStats,
    output_dir: Path,
    *,
    step_stride: int,
    checkpoint_name: str = "student_model_state.pt",
) -> None:
    """Persist the plots and checkpoint produced by ``run_kd_training``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_epoch_plot(stats.epoch_losses, output_dir)
    _save_step_plot(stats.step_losses, output_dir, step_stride)

    save_path = output_dir / checkpoint_name

    if hasattr(student_model, "peft_config"):
        adapter_dir = output_dir / "lora_adapter"
        student_model.save_pretrained(adapter_dir)
        print(f"Saved LoRA adapter weights to {adapter_dir.resolve()}")

        merged_save_path = save_path
        merged_model_dir = output_dir / "merged_student_model"

        try:  # pragma: no cover - depends on peft implementation
            merged_model = student_model.merge_and_unload()
        except AttributeError:
            merged_model = None

        if merged_model is not None:
            torch.save(merged_model.state_dict(), merged_save_path)
            print(
                "Saved merged student model parameters (LoRA applied) to "
                f"{merged_save_path.resolve()}"
            )

            if hasattr(merged_model, "save_pretrained"):
                merged_model.save_pretrained(merged_model_dir)
                print(
                    "Saved merged student model in Hugging Face format to "
                    f"{merged_model_dir.resolve()}"
                )
            return

        print(
            "Warning: Unable to merge LoRA adapters into the base model. "
            "Only adapter weights were saved."
        )

    torch.save(student_model.state_dict(), save_path)
    print(f"Saved student model parameters to {save_path.resolve()}")


__all__ = [
    "DistillationStats",
    "EarlyStoppingConfig",
    "LoraAdapterSettings",
    "LossWeights",
    "apply_lora_adapters",
    "compute_kd_loss",
    "compute_projector_alignment_loss",
    "distillation_step",
    "freeze_student_vision_and_projector",
    "run_kd_training",
    "save_training_artifacts",
    "seed_everything",
]

