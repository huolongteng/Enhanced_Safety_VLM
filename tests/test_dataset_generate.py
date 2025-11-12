from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_stub_modules() -> None:
    if "PIL" not in sys.modules:
        pil_module = types.ModuleType("PIL")

        class _FakeImage:
            def convert(self, *_args, **_kwargs):
                return self

        class _ImageModule:
            def open(self, *_args, **_kwargs):
                return _FakeImage()

        pil_module.Image = _ImageModule()
        sys.modules["PIL"] = pil_module

    if "torch" not in sys.modules:
        torch_module = types.ModuleType("torch")
        torch_utils = types.ModuleType("torch.utils")
        torch_utils_data = types.ModuleType("torch.utils.data")

        class _Dataset:
            pass

        class _DataLoader:
            pass

        torch_utils_data.Dataset = _Dataset
        torch_utils_data.DataLoader = _DataLoader
        torch_utils.data = torch_utils_data
        torch_module.utils = torch_utils

        sys.modules["torch"] = torch_module
        sys.modules["torch.utils"] = torch_utils
        sys.modules["torch.utils.data"] = torch_utils_data

    if "torchvision" not in sys.modules:
        torchvision_module = types.ModuleType("torchvision")
        transforms_module = types.ModuleType("torchvision.transforms")

        class _Compose:
            def __init__(self, steps):
                self.steps = steps

            def __call__(self, value):
                for step in self.steps:
                    value = step(value)
                return value

        transforms_module.Compose = _Compose
        torchvision_module.transforms = transforms_module

        sys.modules["torchvision"] = torchvision_module
        sys.modules["torchvision.transforms"] = transforms_module


_install_stub_modules()

import pytest

from scripts.dataset_generate import DatasetEntry


def _build_mapping(image: str) -> dict:
    return {
        "id": "sample-id",
        "input": {
            "image": image,
            "policy": "Stay safe",
        },
        "output": "{}",
    }


def test_from_mapping_with_posix_path(tmp_path: Path) -> None:
    image_rel = Path("train") / "image.jpg"
    image_path = tmp_path / image_rel
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"binary")

    mapping = _build_mapping(image_rel.as_posix())
    entry = DatasetEntry.from_mapping(mapping, base_dir=tmp_path)

    assert entry.image_path == image_path.resolve()


def test_from_mapping_with_windows_path(tmp_path: Path) -> None:
    image_rel = Path("train") / "image.jpg"
    image_path = tmp_path / image_rel
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"binary")

    mapping = _build_mapping("train\\image.jpg")
    entry = DatasetEntry.from_mapping(mapping, base_dir=tmp_path)

    assert entry.image_path == image_path.resolve()


def test_from_mapping_missing_file(tmp_path: Path) -> None:
    mapping = _build_mapping("missing/image.jpg")

    with pytest.raises(FileNotFoundError) as exc:
        DatasetEntry.from_mapping(mapping, base_dir=tmp_path)

    assert "missing" in str(exc.value)
