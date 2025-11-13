import json
import types
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Minimal stubs for torch and torchvision so the dataset can be imported
# without the full frameworks. These cover only the functionality used in
# ``training_dataset.py``.
# ---------------------------------------------------------------------------
if "torch" not in globals():
    import sys

    torch_module = types.ModuleType("torch")
    utils_module = types.ModuleType("torch.utils")

    class _Dataset:
        """Simple dataset base class mimicking ``torch.utils.data.Dataset``."""

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

    class _DataLoader:
        """Tiny DataLoader implementation sufficient for the tests."""

        def __init__(
            self,
            dataset,
            *,
            batch_size=1,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
            **_kwargs,
        ):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn or (lambda batch: batch)
            self.drop_last = drop_last

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                indices = list(indices)  # Deterministic order for tests.
            batch = []
            for idx in indices:
                batch.append(self.dataset[idx])
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
            if batch and not self.drop_last:
                yield self.collate_fn(batch)

    torch_utils_data_module = types.ModuleType("torch.utils.data")
    torch_utils_data_module.DataLoader = _DataLoader
    torch_utils_data_module.Dataset = _Dataset

    utils_module.data = torch_utils_data_module

    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = utils_module
    sys.modules["torch.utils.data"] = torch_utils_data_module

if "torchvision" not in globals():
    import sys

    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")

    class _Resize:
        def __init__(self, size):
            self.size = size

        def __call__(self, image):
            return image.resize(self.size)

    class _Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image):
            for transform in self.transforms:
                image = transform(image)
            return image

    transforms_module.Resize = _Resize
    transforms_module.Compose = _Compose

    torchvision_module.transforms = transforms_module

    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module

from training_dataset import create_dataloader


class DummyProcessor:
    """Minimal processor stub to exercise dataloader integration."""

    image_token = "<image>"

    def __init__(self):
        self.apply_chat_template_calls = []
        self.forward_calls = []

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        self.apply_chat_template_calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )

        image_tokens = "".join(
            self.image_token
            for message in conversation
            for content in message["content"]
            if content["type"] == "image"
        )
        text_segments = [
            content["text"]
            for message in conversation
            for content in message["content"]
            if content["type"] == "text"
        ]
        return f"{image_tokens}{' '.join(text_segments)}"

    def __call__(self, *, text, images, padding, return_tensors, **kwargs):
        call_record = {
            "text": text,
            "images": images,
            "padding": padding,
            "return_tensors": return_tensors,
            "kwargs": kwargs,
        }
        self.forward_calls.append(call_record)
        return call_record


def _write_image(path: Path, color):
    image = Image.new("RGB", (8, 8), color)
    image.save(path)


def _write_dataset(path: Path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_create_dataloader_produces_batches(tmp_path):
    image_paths = []
    entries = []
    for idx in range(2):
        image_path = tmp_path / f"image_{idx}.jpg"
        _write_image(image_path, (idx * 40, 0, 0))
        image_paths.append(str(image_path))
        entries.append(
            {
                "input": {
                    "image": str(image_path),
                    "policy": f"Policy text {idx}",
                },
                "output": f"Response text {idx}",
            }
        )

    json_path = tmp_path / "dataset.json"
    _write_dataset(json_path, entries)

    processor = DummyProcessor()
    dataloader = create_dataloader(
        json_path,
        processor=processor,
        batch_size=2,
        add_generation_prompt=True,
        padding=False,
        return_tensors="pt",
    )

    batch = next(iter(dataloader))

    assert len(processor.apply_chat_template_calls) == 2
    assert all(call["tokenize"] is False for call in processor.apply_chat_template_calls)
    assert all(call["add_generation_prompt"] is True for call in processor.apply_chat_template_calls)

    assert len(processor.forward_calls) == 1
    forward_call = processor.forward_calls[0]

    assert forward_call["padding"] == "longest"
    assert forward_call["return_tensors"] == "pt"
    assert len(forward_call["text"]) == 2
    assert all(prompt.startswith("<image>") for prompt in forward_call["text"])
    assert len(forward_call["images"]) == 2

    assert batch == forward_call


def test_create_dataloader_raises_for_mismatched_lengths(tmp_path):
    image_a = tmp_path / "image_a.jpg"
    image_b = tmp_path / "image_b.jpg"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    entries = [
        {
            "input": {
                "image": str(image_a),
                "policy": "Policy A",
            },
            "output": "Response A",
        },
        {
            "input": {
                "image": str(image_b),
                "policy": "Policy B",
            },
            "output": "",  # Missing response should trigger a mismatch.
        },
    ]

    json_path = tmp_path / "dataset.json"
    _write_dataset(json_path, entries)

    processor = DummyProcessor()

    with pytest.raises(ValueError):
        create_dataloader(json_path, processor=processor)
