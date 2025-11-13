"""Training helpers for supervised fine-tuning of multimodal models."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence
import matplotlib.pyplot as plt
import torch
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
class TrainingStats:
    """Container for metrics gathered during supervised training.

    Attributes
    ----------
    epoch_losses:
        Average loss value computed at the end of each epoch.
    step_losses:
        Average loss value for each optimizer step (after any gradient
        accumulation).
    best_epoch:
        The 1-based index of the epoch that achieved the lowest observed loss
        when early stopping is enabled.  ``None`` when no improvement was
        recorded or the heuristic is disabled.
    best_loss:
        The best average loss encountered throughout training.  ``None`` when
        the training loop never logged a finite value (e.g. empty dataloader).
    early_stopped:
        ``True`` when the early-stopping heuristic terminated the loop before
        exhausting ``num_epochs``.
    """

    epoch_losses: list[float]
    step_losses: list[float]
    best_epoch: int | None = None
    best_loss: float | None = None
    early_stopped: bool = False


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Parameters that control the optional early-stopping heuristic.

    The defaults follow a conventional setup where training stops after two
    consecutive non-improving epochs while automatically restoring the best
    checkpoint observed so far.
    """

    enabled: bool = False
    patience: int = 2
    min_delta: float = 0.0
    min_epochs: int = 1
    restore_best_weights: bool = True


def seed_everything(seed: int) -> None:
    """Seed all common random number generators."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random

    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _move_batch_to_device(batch, device):
    """Move tensors contained in ``batch`` to ``device`` (labels included)."""

    if device is None:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    # ``BatchEncoding``/``BatchFeature`` objects expose ``to`` and support
    # in-place device transfer for every contained tensor (``labels`` included).
    to_method = getattr(batch, "to", None)
    if callable(to_method):
        try:
            return to_method(device)
        except TypeError:
            # Some containers expose ``to`` with a different signature.  Fall
            # back to item-wise recursion below when that happens.
            pass

    if isinstance(batch, dict):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}

    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_to_device(value, device) for value in batch)

    return batch


def supervised_step(
    student_model,
    optimizer,
    batch,
    *,
    device,
    loss_scale: float = 1.0,
    grad_scaler: "torch.cuda.amp.GradScaler" | None = None,
):
    """Execute a supervised optimisation step for ``student_model``.

    The step performs a forward pass using ``batch`` (which must include
    ``labels`` so that Hugging Face computes the loss) and backpropagates the
    scaled loss.  Optimizer stepping and zero-grad handling remain the
    responsibility of the outer training loop to preserve gradient accumulation
    semantics.
    """

    if optimizer is None:
        raise ValueError("An optimizer instance must be provided for supervised_step().")

    if loss_scale <= 0:
        raise ValueError("loss_scale must be a positive value.")

    model_inputs = _move_batch_to_device(batch, device)

    if optimizer.param_groups:
        first_param = next(
            (
                param
                for group in optimizer.param_groups
                for param in group.get("params", [])
                if param is not None
            ),
            None,
        )
        if first_param is not None and first_param.device != device:
            raise ValueError(
                "Optimizer parameters are not on the target device; make sure the model "
                "has been moved before training."
            )

    outputs = student_model(**model_inputs)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise ValueError(
            "Model outputs did not include a loss. Ensure labels are provided in the batch."
        )

    scaled_loss = loss / float(loss_scale)
    if grad_scaler is not None:
        grad_scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    return float(loss.detach().cpu().item()), outputs


def apply_lora_adapters(
    student_model,
    settings: LoraAdapterSettings | None,
):
    """Wrap ``student_model`` with LoRA adapters when ``settings`` enable it."""

    if settings is None or not settings.enabled:
        return student_model

    try:
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
    """Freeze the vision tower and projector weights of ``student_model``.

    The user workflow only wants to optimize the language-model parameters.
    To respect that requirement we mark every parameter belonging to the
    visual backbone (``vision_tower``) and projector modules as
    ``requires_grad = False``.  This prevents the optimizer from updating the
    weights while keeping the modules available for the forward pass.  The
    function returns the attribute names that were successfully frozen so the
    caller can log them if desired.
    """

    frozen_components: list[str] = []

    # Candidate attribute names observed in LLaVA-style models.  If new
    # variants introduce alternative names, extend this tuple so those modules
    # are frozen automatically as well.
    candidate_attrs = (
        "vision_tower",
        "vision_model",
        "multi_modal_projector",
        "multimodal_projector",
        "vision_projector",
        "mm_projector",
        "projector",
    )

    # Some model wrappers keep the real modules under ``model`` (e.g. ``.model``
    # holds the actual LLaVA implementation).  Iterate over both the wrapper
    # and the nested model to cover each case without making assumptions about
    # the exact class hierarchy.
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

            # If the attribute exposes parameters, disable their gradients.
            parameters = getattr(module, "parameters", None)
            if parameters is None:
                continue

            for param in parameters():
                param.requires_grad = False

            frozen_components.append(attr)

    return frozen_components


def run_supervised_training(
    student_model,
    dataloader: Iterable,
    *,
    device,
    learning_rate: float,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    early_stopping: EarlyStoppingConfig | None = None,
) -> TrainingStats:
    """Train ``student_model`` on labelled batches yielded by ``dataloader``."""

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

    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
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

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        num_batches = 0
        accumulation_loss = 0.0
        accumulated_batches = 0
        current_cycle_size = gradient_accumulation_steps

        progress_bar.set_description(f"Training (epoch {epoch + 1}/{num_epochs})")

        for batch_idx, batch in enumerate(dataloader):
            if accumulated_batches == 0:
                optimizer.zero_grad()
                if steps_per_epoch is not None and steps_per_epoch > 0:
                    remaining_batches = steps_per_epoch - batch_idx
                    current_cycle_size = min(
                        gradient_accumulation_steps,
                        max(1, remaining_batches),
                    )
                else:
                    current_cycle_size = gradient_accumulation_steps

            loss_value, _ = supervised_step(
                student_model,
                optimizer,
                batch,
                device=device,
                loss_scale=float(current_cycle_size),
            )
            running_loss += loss_value
            accumulation_loss += loss_value
            num_batches += 1

            accumulated_batches += 1

            progress_bar.update(1)

            if accumulated_batches >= current_cycle_size:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                average_cycle_loss = accumulation_loss / float(accumulated_batches)
                step_losses.append(average_cycle_loss)
                accumulation_loss = 0.0
                accumulated_batches = 0
                global_step += 1

        if accumulated_batches > 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            average_cycle_loss = accumulation_loss / float(accumulated_batches)
            step_losses.append(average_cycle_loss)
            accumulation_loss = 0.0
            accumulated_batches = 0
            global_step += 1

        epoch_loss = running_loss / max(1, num_batches)
        epoch_losses.append(epoch_loss)

        metrics = [
            f"Epoch {epoch + 1}/{num_epochs}",
            f"average_loss={epoch_loss:.4f}",
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
        ]
        if batch_size is not None:
            metrics.append(f"batch_size={batch_size}")
        metrics.append(f"optimizer_steps={global_step}")

        progress_bar.write(" | ".join(metrics))
        if total_batches is None:
            progress_bar.refresh()

        if early_cfg is not None:
            if epoch_loss + early_cfg.min_delta < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()}
                best_epoch = epoch
                epochs_without_improve = 0
                progress_bar.write(
                    f"New best model found at epoch {epoch + 1} with loss {epoch_loss:.4f}."
                )
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

    return TrainingStats(
        epoch_losses=epoch_losses,
        step_losses=step_losses,
        best_epoch=None if best_epoch is None else best_epoch + 1,
        best_loss=None if best_loss == float('inf') else best_loss,
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
    plt.title("Supervised Training Loss (Epoch)")
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
    plt.title("Supervised Training Loss (Step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "loss_curve_step.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved step loss curve to {plot_path.resolve()}")


def save_training_artifacts(
    student_model,
    stats: TrainingStats,
    output_dir: Path,
    *,
    step_stride: int,
    checkpoint_name: str = "student_model_state.pt",
) -> None:
    """Persist the plots and checkpoint produced by ``run_supervised_training``."""

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

        try:
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

