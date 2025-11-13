"""Dataset utilities for policy-aware, labelled training data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass(frozen=True)
class PolicySample:
    """Container describing a single multimodal policy example."""

    image_path: Path
    policy_text: str
    label_text: str


def _normalise_relative_path(path_value: str) -> Path:
    """Return a POSIX-style relative path for ``path_value``."""

    normalised = path_value.replace("\\", "/")
    return Path(normalised)


def _format_label_text(output_field) -> str:
    """Format the label portion of a dataset entry into natural text."""

    if isinstance(output_field, str):
        try:
            payload = json.loads(output_field)
        except json.JSONDecodeError:
            payload = {"response": output_field}
    elif isinstance(output_field, dict):
        payload = output_field
    else:
        payload = {}

    rating = payload.get("rating")
    category = payload.get("category")
    rationale = payload.get("rationale")

    parts: List[str] = []
    if rating:
        parts.append(f"Rating: {rating}")
    if category:
        parts.append(f"Category: {category}")
    if rationale:
        parts.append(f"Rationale: {rationale}")

    if not parts and output_field is not None:
        parts.append(str(output_field))

    return "\n".join(parts).strip()


def load_split_entries(dataset_json: Path | str, image_root: Path | str) -> List[PolicySample]:
    """Load and normalise entries from ``dataset_json``.

    Parameters
    ----------
    dataset_json:
        Path to the JSON file containing the dataset split.
    image_root:
        Base directory containing the ``train``/``test`` image folders.  The
        argument is accepted for convenience and validation even though the
        returned samples only store relative paths.
    """

    dataset_path = Path(dataset_json)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")

    with dataset_path.open("r", encoding="utf-8") as fp:
        raw_entries = json.load(fp)

    if not isinstance(raw_entries, Sequence):
        raise ValueError("Dataset JSON must contain a list of entries.")

    entries: List[PolicySample] = []
    for idx, entry in enumerate(raw_entries):
        input_payload = entry.get("input", {}) if isinstance(entry, dict) else {}
        image_field = input_payload.get("image")
        policy_text = input_payload.get("policy")

        if image_field is None or policy_text is None:
            raise ValueError(
                f"Entry {idx} in {dataset_path} is missing required input fields."
            )

        relative_path = _normalise_relative_path(str(image_field))
        label_text = _format_label_text(entry.get("output"))

        entries.append(
            PolicySample(
                image_path=relative_path,
                policy_text=str(policy_text),
                label_text=label_text,
            )
        )

    return entries


class PolicyImageDataset(Dataset):
    """Dataset backed by ``PolicySample`` entries and on-disk images."""

    def __init__(
        self,
        entries: Sequence[PolicySample],
        image_root: Path | str,
        image_size: int = 256,
    ) -> None:
        self.entries = list(entries)
        self.image_root = Path(image_root)
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size))])

    def __len__(self) -> int:  # pragma: no cover - trivial container method
        return len(self.entries)

    def __getitem__(self, idx: int):
        sample = self.entries[idx]
        image_path = (self.image_root / sample.image_path).resolve()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample.policy_text},
                ],
            }
        ]

        return {
            "image": image,
            "conversation": conversation,
            "label_text": sample.label_text,
        }


def apply_chat_template_to_batch(
    conversations: Iterable[Sequence[dict]],
    processor,
    add_generation_prompt: bool = False,
) -> List[str]:
    """Apply ``processor.apply_chat_template`` to each conversation."""

    prompts: List[str] = []
    for conv in conversations:
        prompt = processor.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        prompts.append(prompt)
    return prompts


def build_policy_collate_fn(
    processor,
    add_generation_prompt: bool = True,
    padding: bool | str = True,
    return_tensors: str = "pt",
    **processor_kwargs,
):
    """Create a collate function that batches samples with ``processor``."""

    def collate_fn(batch):
        if not batch:
            raise ValueError("Received an empty batch.")

        images = [sample["image"] for sample in batch]
        conversations = [sample["conversation"] for sample in batch]
        targets = [sample["label_text"] for sample in batch]

        prompts = apply_chat_template_to_batch(
            conversations,
            processor,
            add_generation_prompt=add_generation_prompt,
        )

        return processor(
            text=prompts,
            images=images,
            text_target=targets,
            padding=padding,
            return_tensors=return_tensors,
            **processor_kwargs,
        )

    return collate_fn


def create_policy_dataloader(
    entries: Sequence[PolicySample],
    image_root: Path | str,
    processor,
    *,
    batch_size: int,
    image_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Instantiate a ``DataLoader`` backed by the labelled policy dataset."""

    dataset = PolicyImageDataset(entries, image_root=image_root, image_size=image_size)
    collate_fn = build_policy_collate_fn(
        processor,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
