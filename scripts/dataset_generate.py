"""Utility helpers to create DataLoader objects for safety training datasets.

This module mirrors the behaviour of ``.old_verify/dataset.py`` but adapts it to
work with the JSON artefacts produced by :mod:`scripts.training_info_convert`.

The generated JSON files are expected to live under the repository ``data``
directory and follow the structure emitted by ``input_dataset_generate.py`` and
``training_info_convert.py``.  In particular, each entry should contain the
image path (relative to the ``data`` directory), the policy text and the
assistant response string formatted using the required safety label template.

Example entry::

    {
        "id": "0001",
        "input": {
            "image": "train/0001.jpg",
            "policy": "<policy prompt text>"
        },
        "output": "{\n    \"rating\": \"Safe\",\n    \"category\": \"NA: None applying\",\n    \"rationale\": \"...\"\n}\n"
    }

The helpers defined here build ``torch.utils.data.DataLoader`` instances that
can be directly used during training or evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = [
    "DATA_DIR",
    "DatasetEntry",
    "SafetyPolicyDataset",
    "apply_chat_template_to_batch",
    "build_policy_collate_fn",
    "create_policy_dataloader",
    "load_dataset_entries",
    "load_default_dataloaders",
]


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TRAIN_JSON = DATA_DIR / "train_dataset.json"
DEFAULT_EVAL_JSON = DATA_DIR / "test_dataset.json"


@dataclass(slots=True)
class DatasetEntry:
    """Representation of a single dataset example."""

    identifier: str
    image_path: Path
    policy: str
    output: str

    @classmethod
    def from_mapping(
        cls, mapping: MutableMapping[str, Any], *, base_dir: Path
    ) -> "DatasetEntry":
        identifier = str(mapping.get("id") or "").strip()
        input_block = mapping.get("input") or {}
        if not isinstance(input_block, MutableMapping):
            raise TypeError("Expected 'input' to be a mapping in dataset JSON entry")

        image_field = str(input_block.get("image") or "").strip()
        policy_field = str(input_block.get("policy") or "").strip()

        if not image_field:
            raise ValueError("Dataset entry is missing the image path")

        image_path_candidate = Path(image_field)
        if not image_path_candidate.is_absolute():
            image_path = (base_dir / image_path_candidate).resolve()
        else:
            image_path = image_path_candidate
        output_raw = mapping.get("output")
        output_text = output_raw if isinstance(output_raw, str) else str(output_raw or "")

        return cls(
            identifier=identifier,
            image_path=image_path,
            policy=policy_field,
            output=output_text,
        )


def load_dataset_entries(json_path: Path) -> List[DatasetEntry]:
    """Load dataset entries from ``json_path``.

    Parameters
    ----------
    json_path:
        Path to the JSON file produced by ``training_info_convert.py``.
    """

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise TypeError("Dataset JSON must contain a sequence of entries")

    return [DatasetEntry.from_mapping(entry, base_dir=json_path.parent) for entry in data]


class SafetyPolicyDataset(Dataset):
    """Torch dataset that loads image-policy pairs with assistant responses."""

    def __init__(
        self,
        entries: Sequence[DatasetEntry],
        *,
        image_size: int = 256,
    ) -> None:
        self.entries = list(entries)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        image = Image.open(entry.image_path).convert("RGB")
        image = self.transform(image)

        assistant_text = entry.output.strip()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": entry.policy},
                ],
            },
        ]

        if assistant_text:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text},
                    ],
                }
            )

        return {
            "image": image,
            "conversation": conversation,
            "assistant_text": assistant_text,
        }


def apply_chat_template_to_batch(
    conversations: Iterable[Sequence[MutableMapping[str, Any]]],
    processor,
    *,
    add_generation_prompt: bool = False,
) -> List[str]:
    """Apply ``processor.apply_chat_template`` to all conversations in a batch."""

    prompts: List[str] = []
    for conversation in conversations:
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        prompts.append(prompt)
    return prompts


def build_policy_collate_fn(
    processor,
    *,
    add_generation_prompt: bool = False,
    include_labels: bool = False,
    padding: bool | str = False,
    return_tensors: str = "pt",
    return_conversations: bool = False,
    **processor_kwargs: Any,
):
    """Build a collate function that uses ``processor`` for batching."""

    def collate_fn(batch: Sequence[Dict[str, Any]]):
        if not batch:
            raise ValueError("Received an empty batch")

        images = [sample["image"] for sample in batch]
        conversations = [sample["conversation"] for sample in batch]

        processor_kwargs.setdefault("truncation", True)

        if include_labels:
            # Prepare two flavours of the conversation text: one including the
            # assistant response and another limited to the user turn.  The
            # former is used to build ``input_ids`` while the latter provides
            # the prompt length so that the corresponding label tokens can be
            # masked out.
            full_texts = apply_chat_template_to_batch(
                conversations,
                processor,
                add_generation_prompt=False,
            )

            user_only_conversations: List[Sequence[MutableMapping[str, Any]]] = []
            for conversation in conversations:
                if conversation and conversation[-1].get("role") == "assistant":
                    user_only_conversations.append(conversation[:-1])
                else:
                    user_only_conversations.append(conversation)

            prompt_texts = apply_chat_template_to_batch(
                user_only_conversations,
                processor,
                add_generation_prompt=True,
            )

            model_inputs = processor(
                text=full_texts,
                images=images,
                padding=padding,
                return_tensors=return_tensors,
                **processor_kwargs,
            )

            prompt_tokens = processor.tokenizer(
                prompt_texts,
                padding=padding,
                return_tensors=return_tensors,
                truncation=True,
            )

            labels = model_inputs["input_ids"].clone()
            attention_mask = prompt_tokens.get("attention_mask")
            pad_id = getattr(processor.tokenizer, "pad_token_id", None)

            if pad_id is not None:
                labels[labels == pad_id] = -100

            original_batch = labels.size(0)
            valid_indices: List[int] = []

            for idx in range(original_batch):
                if attention_mask is not None:
                    prompt_length = int(attention_mask[idx].sum().item())
                else:
                    row = prompt_tokens["input_ids"][idx]
                    if pad_id is not None:
                        prompt_length = int((row != pad_id).sum().item())
                    else:
                        prompt_length = len(row)

                labels[idx, :prompt_length] = -100

                if (labels[idx] != -100).any():
                    valid_indices.append(idx)

            if len(valid_indices) != original_batch:
                if not valid_indices:
                    raise ValueError(
                        "All samples in the batch were filtered out because they "
                        "did not contain assistant target tokens."
                    )

                index_tensor = torch.tensor(valid_indices, device=labels.device, dtype=torch.long)
                labels = labels.index_select(0, index_tensor)

                for key, value in list(model_inputs.items()):
                    if key == "labels":
                        continue

                    if isinstance(value, torch.Tensor) and value.size(0) == original_batch:
                        model_inputs[key] = value.index_select(0, index_tensor.to(value.device))
                    elif isinstance(value, list) and len(value) == original_batch:
                        model_inputs[key] = [value[i] for i in valid_indices]

                conversations[:] = [conversations[i] for i in valid_indices]

            model_inputs["labels"] = labels
        else:
            prompt_texts = apply_chat_template_to_batch(
                conversations,
                processor,
                add_generation_prompt=add_generation_prompt,
            )

            model_inputs = processor(
                text=prompt_texts,
                images=images,
                padding=padding,
                return_tensors=return_tensors,
                **processor_kwargs,
            )

        if return_conversations:
            model_inputs["conversations"] = conversations

        return model_inputs

    return collate_fn


def create_policy_dataloader(
    entries: Sequence[DatasetEntry],
    processor,
    *,
    batch_size: int,
    image_size: int,
    shuffle: bool = True,
    include_labels: bool = True,
) -> DataLoader:
    """Create a dataloader from dataset entries and a processor."""

    dataset = SafetyPolicyDataset(entries, image_size=image_size)
    collate_fn = build_policy_collate_fn(
        processor,
        add_generation_prompt=True,
        include_labels=include_labels,
        padding=True,
        return_tensors="pt",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def load_default_dataloaders(
    processor,
    *,
    batch_size: int,
    image_size: int,
    shuffle_train: bool = True,
    include_labels: bool = True,
) -> Dict[str, DataLoader]:
    """Load train and evaluation dataloaders using default JSON paths."""

    train_entries = load_dataset_entries(DEFAULT_TRAIN_JSON)
    eval_entries = load_dataset_entries(DEFAULT_EVAL_JSON)

    train_loader = create_policy_dataloader(
        train_entries,
        processor,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle_train,
        include_labels=include_labels,
    )
    eval_loader = create_policy_dataloader(
        eval_entries,
        processor,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        include_labels=include_labels,
    )

    return {"train": train_loader, "eval": eval_loader}


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(description="Preview dataset sizes for sanity checks")
    parser.add_argument("processor", help="Dotted import path to the processor object")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for preview")
    parser.add_argument("--image-size", type=int, default=256, help="Image resize dimension")
    args = parser.parse_args()

    module_path, _, attribute = args.processor.rpartition(":")
    if not module_path:
        module_path, attribute = args.processor.rsplit(".", 1)

    module = __import__(module_path, fromlist=[attribute])
    processor_obj = getattr(module, attribute)

    loaders = load_default_dataloaders(
        processor_obj,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} samples")
