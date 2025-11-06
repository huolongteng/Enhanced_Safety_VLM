"""Training helpers for the knowledge-distillation workflow."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

@dataclass
class DistillationStats:
    """Container for metrics gathered during training."""

    epoch_losses: list[float]
    step_losses: list[float]


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
    """Move every tensor in batch to device."""
    if device is None:
        return batch
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


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


def forward_teacher_student(
    teacher_model,
    student_model,
    batch,
    device,
):
    """Run a forward pass for both teacher and student on ``batch``."""
    model_inputs = _move_batch_to_device(batch, device)

    with torch.no_grad():
        teacher_outputs = teacher_model(**model_inputs)

    student_outputs = student_model(**model_inputs)
    return teacher_outputs, student_outputs, model_inputs


def compute_kd_loss(
    student_logits,
    teacher_logits,
    attention_mask,
    temperature=1.0,
):
    """Compute the standard KL-divergence loss for knowledge distillation."""

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

    # Align sequence length if they differ.
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
) -> float:
    """Perform a single optimization step of knowledge distillation."""

    teacher_outputs, student_outputs, model_inputs = forward_teacher_student(
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
        temperature=temperature,
    )

    total_loss = kd_loss
    projector_loss = None
    if projector_loss_weight > 0.0:
        projector_loss = compute_projector_alignment_loss(
            student_outputs,
            teacher_outputs,
        )
        if projector_loss is not None:
            mix = max(0.0, min(1.0, projector_loss_weight))
            total_loss = (1.0 - mix) * kd_loss + mix * projector_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return float(total_loss.item())


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

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    scheduler = None

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    global_step = 0

    batch_size = getattr(dataloader, "batch_size", None)

    steps_per_epoch = None
    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
        steps_per_epoch = None

    total_steps = None
    if steps_per_epoch is not None and steps_per_epoch > 0:
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        unit="step" if total_steps is not None else "batch",
        dynamic_ncols=True,
        leave=True,
    )

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        num_batches = 0

        progress_bar.set_description(f"Training (epoch {epoch + 1}/{num_epochs})")

        for batch in dataloader:
            loss_value = distillation_step(
                teacher_model,
                student_model,
                optimizer,
                batch,
                device=device,
                temperature=temperature,
                projector_loss_weight=projector_loss_weight,
            )
            running_loss += loss_value
            num_batches += 1
            global_step += 1
            step_losses.append(loss_value)
            progress_bar.update(1)

            if scheduler is not None:
                scheduler.step()

        average_loss = running_loss / max(num_batches, 1)
        epoch_losses.append(average_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        metrics = [
            f"Epoch {epoch + 1}/{num_epochs}",
            f"average_loss={average_loss:.4f}",
            f"lr={current_lr:.6g}",
            f"temperature={temperature:.2f}",
        ]
        if batch_size is not None:
            metrics.append(f"batch_size={batch_size}")
        metrics.append(f"steps={global_step}")

        progress_bar.write(" | ".join(metrics))
        if total_steps is None:
            progress_bar.refresh()

    progress_bar.close()

    return DistillationStats(epoch_losses=epoch_losses, step_losses=step_losses)


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
    torch.save(student_model.state_dict(), save_path)
    print(f"Saved student model parameters to {save_path.resolve()}")

