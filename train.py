"""Training helpers for the knowledge-distillation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def distillation_step(
    teacher_model,
    student_model,
    optimizer,
    batch,
    *,
    device,
    temperature: float,
) -> float:
    """Perform a single optimization step of knowledge distillation."""

    teacher_outputs, student_outputs, model_inputs = forward_teacher_student(
        teacher_model,
        student_model,
        batch,
        device,
    )

    attention_mask = model_inputs.get("attention_mask")
    loss = compute_kd_loss(
        student_outputs.logits,
        teacher_outputs.logits,
        attention_mask,
        temperature=temperature,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def run_kd_training(
    teacher_model,
    student_model,
    dataloader: Iterable,
    *,
    device,
    learning_rate: float,
    temperature: float,
    num_epochs: int,
) -> DistillationStats:
    """Train ``student_model`` to mimic ``teacher_model`` using ``dataloader``."""

    teacher_model.eval()
    teacher_model.to(device)
    student_model.to(device)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    global_step = 0

    batch_size = getattr(dataloader, "batch_size", None)

    epoch_progress = tqdm(
        total=num_epochs,
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            loss_value = distillation_step(
                teacher_model,
                student_model,
                optimizer,
                batch,
                device=device,
                temperature=temperature,
            )
            running_loss += loss_value
            num_batches += 1
            global_step += 1
            step_losses.append(loss_value)

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

        epoch_progress.write(" | ".join(metrics))
        epoch_progress.update(1)

    epoch_progress.close()

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

