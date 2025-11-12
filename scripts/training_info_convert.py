"""Generate training datasets that pair images with policy-aligned labels.

The script mirrors the high level workflow from ``.old_verify/dataset.py`` but
streamlines it to match the latest requirements:

1. Discover the usable images inside ``data/train`` and ``data/test`` using the
   helper from :mod:`scripts.check_image_usable`.
2. Look up the matching annotation row in ``data/<split>.csv`` using the image
   identifier.
3. Combine the policy prompt (model input) and the safety assessment (model
   output) into a simple JSON structure.
4. Save the resulting samples into ``data/<split>_dataset.json``.

Each dataset entry is stored as a JSON object with the following layout::

    {
        "id": "<image identifier>",
        "input": {
            "image": "<path relative to the data directory>",
            "policy": "<policy prompt text>"
        },
        "output": "{\n    \"rating\": \"...\",\n    \"category\": \"...\",\n    \"rationale\": \"...\"\n}\n"
    }

The output string keeps the exact formatting (including the trailing newline)
requested by the user so that it can be fed directly to instruction-following
models during training.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

try:  # Pillow is required by ``check_image_usable`` to open images.
    from check_image_usable import available_images
except SystemExit as exc:  # pragma: no cover - handled during runtime
    raise SystemExit(
        "Pillow is required to generate the dataset. Install it with 'pip install pillow'."
    ) from exc


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SPLITS: tuple[str, ...] = ("train", "test")


def load_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    """Yield the annotation rows stored in ``csv_path``."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            yield row


def build_row_index(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Create a mapping from image identifier to CSV row."""

    index: Dict[str, Dict[str, str]] = {}
    for row in rows:
        identifier = (row.get("id") or "").strip()
        if not identifier:
            continue
        index.setdefault(identifier, row)
    return index


def label_to_template(row: Dict[str, str]) -> str:
    """Format the label portion of a row using the requested template."""

    rating = (row.get("rating") or "").strip()
    category = (row.get("category") or "").strip()
    rationale = (row.get("rationale") or "").strip()

    return (
        "{\n"
        f"    \"rating\": {json.dumps(rating)},\n"
        f"    \"category\": {json.dumps(category)},\n"
        f"    \"rationale\": {json.dumps(rationale)}\n"
        "}\n"
    )


def relative_image_path(image_path: Path, base_dir: Path) -> str:
    """Return ``image_path`` relative to ``base_dir`` when possible."""

    try:
        return str(image_path.relative_to(base_dir))
    except ValueError:
        return str(image_path)


def build_dataset_for_split(split: str) -> List[Dict[str, object]]:
    """Create the dataset for ``split`` (``train`` or ``test``)."""

    images_dir = DATA_DIR / split
    if not images_dir.exists():
        print(f"Warning: images directory not found for split '{split}': {images_dir}")
        return []

    usable_images = available_images(images_dir)
    if not usable_images:
        print(f"Warning: no usable images found for split '{split}'.")
        return []

    csv_path = DATA_DIR / f"{split}.csv"
    row_index = build_row_index(load_rows(csv_path))

    dataset: List[Dict[str, object]] = []
    skipped = 0

    for image_path in usable_images:
        identifier = image_path.stem
        row = row_index.get(identifier)
        if row is None:
            skipped += 1
            continue

        policy_text = (row.get("policy") or "").strip()
        entry = {
            "id": identifier,
            "input": {
                "image": relative_image_path(image_path, DATA_DIR),
                "policy": policy_text,
            },
            "output": label_to_template(row),
        }
        dataset.append(entry)

    if skipped:
        print(
            f"Split '{split}': {skipped} usable images skipped because no CSV row matched."
        )

    print(f"Split '{split}': generated {len(dataset)} samples.")
    return dataset


def write_split_dataset(split: str, samples: List[Dict[str, object]]) -> None:
    """Write ``samples`` into ``data/<split>_dataset.json``."""

    if not samples:
        print(f"Split '{split}': nothing to write.")
        return

    output_path = DATA_DIR / f"{split}_dataset.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Split '{split}': wrote {output_path.relative_to(DATA_DIR.parent)}")


def main() -> int:
    for split in DEFAULT_SPLITS:
        samples = build_dataset_for_split(split)
        write_split_dataset(split, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
