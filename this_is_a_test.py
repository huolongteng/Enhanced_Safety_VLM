import copy
import json
import types
from pathlib import Path

import pytest

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal environments.
    import sys

    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")
    _STORED_IMAGES = {}

    class _FakeImage:
        def __init__(self, mode, size, color):
            self.mode = mode
            self.size = size
            self.color = color

        def resize(self, size):
            return _FakeImage(self.mode, size, self.color)

        def convert(self, _mode):
            return self

        def copy(self):
            return _FakeImage(self.mode, self.size, self.color)

        def save(self, path):
            _STORED_IMAGES[str(path)] = self.copy()

    def _new(mode, size, color):
        return _FakeImage(mode, size, color)

    def _open(path):
        try:
            return _STORED_IMAGES[str(path)].copy()
        except KeyError as exc:  # pragma: no cover - defensive.
            raise FileNotFoundError(str(path)) from exc

    pil_image_module.new = _new
    pil_image_module.open = _open

    pil_module.Image = pil_image_module

    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image_module

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


class DummyTensor:
    """Tiny tensor-like object mimicking the pieces of ``torch.Tensor`` we need."""

    def __init__(self, data):
        self.data = data

    def clone(self):
        return DummyTensor(copy.deepcopy(self.data))

    def size(self, dim=None):
        if dim is None:
            return (len(self.data), len(self.data[0]) if self.data else 0)
        if dim == 0:
            return len(self.data)
        if dim == 1:
            return len(self.data[0]) if self.data else 0
        raise ValueError("DummyTensor only supports two dimensions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            if isinstance(col, slice):
                return self.data[row][col]
            return self.data[row][col]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            if isinstance(col, slice):
                indices = range(*col.indices(len(self.data[row])))
                for idx in indices:
                    self.data[row][idx] = value
                return
            self.data[row][col] = value
            return
        self.data[key] = value

    def masked_fill(self, mask, value):
        result = self.clone()
        for row_idx in range(len(mask)):
            row_mask = mask[row_idx]
            for col_idx, should_fill in enumerate(row_mask):
                if should_fill:
                    result[row_idx, col_idx] = value
        return result

    def tolist(self):
        return copy.deepcopy(self.data)

    def __iter__(self):
        return iter(self.data)

    def __eq__(self, other):
        return DummyTensor([[value == other for value in row] for row in self.data])


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.vocab = {
            "<bos>": 1,
            "<image>": 2,
            "USER:": 3,
            "ASSISTANT:": 4,
            "<eos>": 5,
        }

    def _tokenize(self, text):
        return text.replace("\n", " ").split()

    def _convert(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def __call__(self, text, add_special_tokens):
        tokens = self._tokenize(text)
        return {"input_ids": [self._convert(token) for token in tokens]}


class DummyProcessor:
    """Minimal processor stub to exercise dataloader integration."""

    image_token = "<image>"

    def __init__(self):
        self.apply_chat_template_calls = []
        self.forward_calls = []
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        self.apply_chat_template_calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )

        image_tokens = " ".join(
            self.image_token
            for message in conversation
            for content in message["content"]
            if content["type"] == "image"
        )
        user_texts = [
            content["text"]
            for message in conversation
            if message["role"] == "user"
            for content in message["content"]
            if content["type"] == "text"
        ]
        assistant_texts = [
            content["text"]
            for message in conversation
            if message["role"] == "assistant"
            for content in message["content"]
            if content["type"] == "text"
        ]

        if add_generation_prompt:
            return f"<bos> {image_tokens} USER: {user_texts[0]} ASSISTANT:"

        assistant_segment = assistant_texts[0] if assistant_texts else ""
        return f"<bos> {image_tokens} USER: {user_texts[0]} ASSISTANT: {assistant_segment} <eos>"

    def __call__(self, *, text, images, padding, return_tensors, **kwargs):
        sequences = [self.tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in text]
        max_length = max(len(seq) for seq in sequences)

        padded = [seq + [self.tokenizer.pad_token_id] * (max_length - len(seq)) for seq in sequences]
        attention = [
            [1] * len(seq) + [0] * (max_length - len(seq))
            for seq in sequences
        ]

        pixel_values = [
            [[[0 for _ in range(4)] for _ in range(4)] for _ in range(3)]
            for _ in range(len(images))
        ]

        call_record = {
            "text": text,
            "images": images,
            "padding": padding,
            "return_tensors": return_tensors,
            "kwargs": kwargs,
            "input_ids": DummyTensor(padded),
            "attention_mask": DummyTensor(attention),
            "pixel_values": DummyTensor(pixel_values),
        }
        self.forward_calls.append(call_record)
        return call_record


def _write_image(path: Path, color):
    image = Image.new("RGB", (8, 8), color)
    image.save(path)


def _write_dataset(path: Path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_create_dataloader_generation_prompts(tmp_path):
    entries = []
    for idx in range(2):
        image_path = tmp_path / f"image_{idx}.jpg"
        _write_image(image_path, (idx * 40, 0, 0))
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

    forward_call = processor.forward_calls[0]
    assert batch is forward_call
    assert forward_call["padding"] == "longest"
    assert forward_call["return_tensors"] == "pt"
    assert len(forward_call["text"]) == 2
    assert all(prompt.startswith("<bos> <image>") for prompt in forward_call["text"])
    assert len(forward_call["images"]) == 2


def test_supervised_labels_mask_user_and_padding(tmp_path):
    image_path = tmp_path / "image.jpg"
    _write_image(image_path, (128, 0, 0))

    json_path = tmp_path / "dataset.json"
    _write_dataset(
        json_path,
        [
            {
                "input": {
                    "image": str(image_path),
                    "policy": "Inspect the item",
                },
                "output": "Return classification",
            }
        ],
    )

    processor = DummyProcessor()
    dataloader = create_dataloader(
        json_path,
        processor=processor,
        batch_size=1,
        add_generation_prompt=False,
        padding=False,
        return_tensors="pt",
    )

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].tolist()[0]
    labels = batch["labels"].tolist()[0]

    # The dummy template produces: <bos> <image> USER: Inspect the item ASSISTANT: Return classification <eos>
    assert input_ids[:6] == [1, 2, 3, 6, 7, 8]
    # User segment (everything up to "ASSISTANT:" token) is masked.
    assistant_index = input_ids.index(processor.tokenizer.vocab["ASSISTANT:"])
    assert all(label == -100 for label in labels[: assistant_index + 1])
    # Assistant content remains for loss computation until padding kicks in.
    assert labels[assistant_index + 1] != -100
    # Padding (if any) is masked out.
    attention = batch["attention_mask"].tolist()[0]
    for idx, mask in enumerate(attention):
        if mask == 0:
            assert labels[idx] == -100


def test_invalid_records_are_skipped(tmp_path):
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

    dataloader = create_dataloader(json_path, processor=processor, batch_size=1)
    batch = next(iter(dataloader))

    # Only the first valid record should remain in the dataset (two template
    # calls for the surviving sample: prompt + generation prompt mask).
    assert len(processor.apply_chat_template_calls) == 2
    assert batch["text"][0].endswith("Response A <eos>")
