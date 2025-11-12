"""Utility for checking dataset images.

This lightweight script inspects the images stored inside ``data/train`` and
``data/test``.  It verifies that each file can be opened by Pillow and returns
only the paths of the images that are confirmed to be readable.  A short
summary is printed for every directory, reporting how many files could not be
opened.

The first step of the wider dataset preparation pipeline simply needs to know
which images are usable.  Additional processing (such as handling the CSV
files) will be added later once the requirements are finalised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Pillow is required to run this script. Install it with 'pip install pillow'."
    ) from exc


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def is_supported_image(path: Path) -> bool:
    """Return ``True`` if ``path`` has a supported image extension."""

    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def verify_image(path: Path) -> bool:
    """Return ``True`` when the file at ``path`` can be opened by Pillow."""

    try:
        with Image.open(path) as image:
            image.verify()
        # ``verify`` leaves the file in an unusable state for further operations,
        # so we reopen and fully load it to ensure the image data is intact.
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError):
        return False
    return True


def available_images(images_dir: Path) -> List[Path]:
    """Return the paths of images inside ``images_dir`` that can be opened."""

    valid: List[Path] = []
    broken = 0

    for file_path in sorted(images_dir.iterdir()):
        if not file_path.is_file() or not is_supported_image(file_path):
            continue
        if verify_image(file_path):
            valid.append(file_path)
        else:
            broken += 1

    print(f"{images_dir}: unusable images = {broken}")
    return valid


def dataset_directories(root: Path) -> Iterable[Path]:
    """Yield the default dataset directories (train and test)."""

    for split in ("train", "test"):
        candidate = root / split
        if candidate.exists():
            yield candidate
        else:
            print(f"Warning: directory not found: {candidate}")


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "data"

    for directory in dataset_directories(root):
        usable = available_images(directory)
        print(f"Found {len(usable)} usable images in {directory}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
