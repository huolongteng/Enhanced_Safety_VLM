# This block is about kd training process.
import torch
import torch.nn.functional as F

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
    """Run a forward pass for both teacher and student on ``batch``.
    """
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

