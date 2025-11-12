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
error) it will be skipped gracefully. The downloader rotates realistic
browser headers, retries transient HTTP errors with exponential backoff, and
supports optional randomized delays between requests to reduce the likelihood
of triggering anti-scraping systems.

For gated resources on Hugging Face, you must supply an access token that
starts with ``hf_`` (for example, ``hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx``).
There are three interchangeable ways to provide the token:

* Run ``huggingface-cli login`` once. The credential is stored in the local
  Hugging Face cache and will be picked up automatically.
* Export the ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN``) environment variable
  before running this script, for example::

      export HF_TOKEN="hf_your_personal_access_token"
      python scripts/download_images.py

* Pass the token explicitly with ``--token`` when invoking the script::

      python scripts/download_images.py --token hf_your_personal_access_token

When a valid token is present the script automatically authenticates the
``requests`` session so gated downloads succeed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from huggingface_hub import HfFolder
except ImportError:  # pragma: no cover - optional dependency
    HfFolder = None  # type: ignore


CHUNK_SIZE = 8192
DEFAULT_TIMEOUT = 20
DEFAULT_DATASETS = ("train", "test")

# A small pool of modern browser user-agents to rotate between so requests do not
# all share the same easily blocked crawler signature. The list intentionally
# mirrors popular desktop browsers.
ROTATING_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
)

# Commonly accepted headers to accompany the randomized user-agent and better
# mimic typical browser requests.
BASE_REQUEST_HEADERS: Dict[str, str] = {
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def configure_logging() -> None:
    """Configure basic logging for the script."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def _build_request_headers(url: str) -> Dict[str, str]:
    """Create per-request headers with a rotated user-agent and referer."""

    headers = dict(BASE_REQUEST_HEADERS)
    headers["User-Agent"] = random.choice(ROTATING_USER_AGENTS)

    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        referer = f"{parsed.scheme}://{parsed.netloc}"
        headers["Referer"] = referer

    return headers


def _normalize_extension(extension: str) -> Optional[str]:
    """Normalize common image extensions to a canonical lowercase form."""

    if not extension:
        return None

    extension = extension.lower()
    if extension == ".jpeg":
        return ".jpg"

    if extension in {".jpg", ".png"}:
        return extension

    return None


def _determine_destination(output_dir: Path, image_id: str, url: str) -> Optional[Path]:
    """Derive the destination path for a download based on the URL extension."""

    parsed = urlsplit(url)
    extension = _normalize_extension(Path(parsed.path).suffix)

    if not extension:
        logging.warning(
            "Unable to determine image extension for %s (id=%s); skipping.", url, image_id
        )
        return None

    return output_dir / f"{image_id}{extension}"


def download_image(
    session: requests.Session,
    url: str,
    destination: Path,
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """Download a single image.

    Args:
        session: An existing :class:`requests.Session` for HTTP requests.
        url: The URL of the image to download.
        destination: The path where the image should be stored.

    Returns:
        ``True`` if the download succeeds, ``False`` otherwise.
    """

    try:
        response = session.get(
            url,
            timeout=timeout,
            stream=True,
            headers=_build_request_headers(url),
        )
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


def resolve_auth_token(override: Optional[str] = None) -> Optional[str]:
    """Return an access token for gated resources if available."""

    if override:
        return override.strip() or None

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token.strip() or None

    if HfFolder is not None:  # pragma: no branch - simple retrieval
        try:
            stored_token = HfFolder.get_token()
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("Unable to load Hugging Face token: %s", exc)
        else:
            if stored_token:
                return stored_token.strip() or None

    return None


def create_session(token: Optional[str] = None) -> requests.Session:
    """Create an HTTP session configured for dataset downloads."""

    session = requests.Session()

    token = resolve_auth_token(token)
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        session.cookies.set("token", token, domain="huggingface.co")
        logging.info("Using Hugging Face authentication token for downloads.")

    retry = Retry(
        total=4,
        status=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def process_dataset(
    csv_path: Path,
    output_dir: Path,
    failure_records: List[Dict[str, str]],
    token: Optional[str] = None,
    *,
    min_delay: float = 0.0,
    max_delay: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[int, int]:
    """Download all images defined in a CSV dataset.

    Args:
        csv_path: Path to the CSV file containing the dataset entries.
        output_dir: Directory where downloaded images should be stored.

    Returns:
        A tuple ``(downloaded, skipped)`` with the number of successfully
        downloaded images and the number of entries skipped due to errors,
        missing data, or repeated files.
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

        with create_session(token) as session:
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

                destination = _determine_destination(output_dir, image_id, url)

                if destination is None:
                    skipped += 1
                    failure_records.append(
                        {
                            "csv_file": csv_path.name,
                            "id": image_id,
                            "url": url,
                        }
                    )
                    continue

                if destination.exists():
                    logging.info("Image already exists, skipping: %s", destination)
                    skipped += 1
                    continue

                if download_image(session, url, destination, timeout=timeout):
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

                if max_delay > 0:
                    delay = random.uniform(min_delay, max_delay)
                    if delay > 0:
                        time.sleep(delay)

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
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face access token for authenticated downloads. Overrides "
            "environment variables and stored credentials."
        ),
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.0,
        help=(
            "Minimum delay in seconds between downloads. Use together with --max-delay "
            "to introduce random pauses that mimic human browsing patterns."
        ),
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.0,
        help=(
            "Maximum delay in seconds between downloads. Values greater than zero "
            "enable randomized throttling."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds. Increase if downloading large files.",
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


def main(
    datasets: Iterable[str],
    data_dir: Path,
    failure_log: Path,
    token: Optional[str] = None,
    *,
    min_delay: float = 0.0,
    max_delay: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    min_delay = max(min_delay, 0.0)
    max_delay = max(max_delay, 0.0)

    if max_delay and max_delay < min_delay:
        logging.warning(
            "Max delay %.2fs is smaller than min delay %.2fs; swapping values.",
            max_delay,
            min_delay,
        )
        min_delay, max_delay = max_delay, min_delay

    if min_delay > 0 and max_delay == 0:
        max_delay = min_delay

    total_downloaded = 0
    total_skipped = 0
    failure_records: List[Dict[str, str]] = []

    for dataset in datasets:
        csv_path = data_dir / f"{dataset}.csv"
        output_dir = data_dir / dataset

        logging.info("Processing dataset '%s'", dataset)
        downloaded, skipped = process_dataset(
            csv_path,
            output_dir,
            failure_records,
            token,
            min_delay=min_delay,
            max_delay=max_delay,
            timeout=timeout,
        )
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
    main(
        args.datasets,
        args.data_dir,
        failure_log,
        args.token,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        timeout=args.timeout,
    )
