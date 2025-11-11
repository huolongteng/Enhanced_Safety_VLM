"""Download images referenced in the dataset CSV files.

This script reads the ``train.csv`` and ``test.csv`` files located in the
``data`` directory. Each CSV file is expected to contain at least the
``id`` and ``url`` columns. Images are downloaded from the URLs and stored
in ``data/train`` and ``data/test`` respectively, using the ``id`` value as
the filename.

Example usage::

    python scripts/download_images.py

    # Download only the train split
    python scripts/download_images.py --datasets train

If an image cannot be downloaded (e.g. due to an invalid URL or network
error) it will be skipped gracefully.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


CHUNK_SIZE = 8192
DEFAULT_DATASETS = ("train", "test")


def configure_logging() -> None:
    """Configure basic logging for the script."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def download_image(session: requests.Session, url: str, destination: Path) -> bool:
    """Download a single image.

    Args:
        session: An existing :class:`requests.Session` for HTTP requests.
        url: The URL of the image to download.
        destination: The path where the image should be stored.

    Returns:
        ``True`` if the download succeeds, ``False`` otherwise.
    """

    try:
        response = session.get(url, timeout=15, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.warning("Failed to download %s: %s", url, exc)
        return False

    # Ensure the parent directory exists before writing the file.
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
    except OSError as exc:
        logging.error("Unable to write image to %s: %s", destination, exc)
        return False

    return True


def process_dataset(
    csv_path: Path,
    output_dir: Path,
    failure_records: List[Dict[str, str]],
) -> Tuple[int, int]:
    """Download all images defined in a CSV dataset.

    Args:
        csv_path: Path to the CSV file containing the dataset entries.
        output_dir: Directory where downloaded images should be stored.

    Returns:
        A tuple ``(downloaded, skipped)`` with the number of successfully
        downloaded images and the number of entries skipped due to
        errors or missing data.
    """

    if not csv_path.exists():
        logging.error("CSV file not found: %s", csv_path)
        return (0, 0)

    downloaded = 0
    skipped = 0

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        # Validate that required columns exist.
        if "id" not in reader.fieldnames or "url" not in reader.fieldnames:
            logging.error("CSV file %s is missing required columns.", csv_path)
            return (0, 0)

        with requests.Session() as session:
            session.headers.update({"User-Agent": "EnhancedSafetyVLM/1.0"})

            for row in reader:
                image_id = (row.get("id") or "").strip()
                url = (row.get("url") or "").strip()

                if not image_id or not url:
                    logging.warning(
                        "Skipping row with missing id/url in %s: %s", csv_path.name, row
                    )
                    skipped += 1
                    failure_records.append(
                        {
                            "csv_file": csv_path.name,
                            "id": image_id or row.get("id", ""),
                            "url": url or row.get("url", ""),
                        }
                    )
                    continue

                destination = output_dir / image_id

                if destination.exists():
                    logging.info("Image already exists, skipping: %s", destination)
                    skipped += 1
                    continue

                if download_image(session, url, destination):
                    downloaded += 1
                else:
                    skipped += 1
                    failure_records.append(
                        {
                            "csv_file": csv_path.name,
                            "id": image_id,
                            "url": url,
                        }
                    )

    return downloaded, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=(
            "Datasets to download (default: train test). Each dataset must have a "
            "corresponding CSV file in the data directory."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=Path(__file__).resolve().parents[1] / "data",
        type=Path,
        help="Root directory containing the CSV files and where images will be stored.",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help=(
            "Optional path to a CSV file where failed downloads will be recorded. "
            "Defaults to data/failed_downloads.csv inside the data directory."
        ),
    )
    return parser.parse_args()


def write_failure_log(failure_log: Path, records: List[Dict[str, str]]) -> None:
    """Persist failed download information to disk."""

    if not records:
        logging.info("No failed downloads detected.")
        return

    failure_log.parent.mkdir(parents=True, exist_ok=True)
    with failure_log.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("csv_file", "id", "url"))
        writer.writeheader()
        writer.writerows(records)

    logging.info("Recorded %d failed downloads to %s", len(records), failure_log)


def main(datasets: Iterable[str], data_dir: Path, failure_log: Path) -> None:
    total_downloaded = 0
    total_skipped = 0
    failure_records: List[Dict[str, str]] = []

    for dataset in datasets:
        csv_path = data_dir / f"{dataset}.csv"
        output_dir = data_dir / dataset

        logging.info("Processing dataset '%s'", dataset)
        downloaded, skipped = process_dataset(csv_path, output_dir, failure_records)
        total_downloaded += downloaded
        total_skipped += skipped

        logging.info(
            "Finished dataset '%s': %d downloaded, %d skipped",
            dataset,
            downloaded,
            skipped,
        )

    logging.info(
        "All datasets complete: %d downloaded, %d skipped", total_downloaded, total_skipped
    )
    write_failure_log(failure_log, failure_records)


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    failure_log = args.failure_log or (args.data_dir / "failed_downloads.csv")
    main(args.datasets, args.data_dir, failure_log)
