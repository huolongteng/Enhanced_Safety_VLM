"""Check CSV files for uniqueness of the `id` column.

This script inspects the `train.csv` and `test.csv` files under the
project's `data` directory and verifies that the values in their `id`
columns are unique.  It prints a short report for each file indicating
whether duplicates were found.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import csv


def read_column_values(csv_path: Path, column_name: str) -> Iterable[str]:
    """Yield values from ``column_name`` in ``csv_path``.

    Parameters
    ----------
    csv_path:
        Path to the CSV file to read.
    column_name:
        Name of the column whose values should be yielded.

    Raises
    ------
    ValueError
        If the requested column is missing from the CSV file.
    """

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or column_name not in reader.fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found in {csv_path.name}."
            )
        for row in reader:
            yield row[column_name]


def check_uniqueness(csv_path: Path, column_name: str = "id") -> bool:
    """Check that ``column_name`` values in ``csv_path`` are unique.

    Returns ``True`` when all values are unique, otherwise ``False``.
    """

    values = list(read_column_values(csv_path, column_name))
    counter = Counter(values)
    duplicates = {value: count for value, count in counter.items() if count > 1}

    if duplicates:
        print(f"{csv_path.name}: Found duplicates in '{column_name}' column.")
        for value, count in duplicates.items():
            print(f"  Value '{value}' appears {count} times")
        return False

    print(f"{csv_path.name}: All values in '{column_name}' column are unique.")
    return True


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    csv_files = ["train.csv", "test.csv"]

    success = True
    for file_name in csv_files:
        csv_path = data_dir / file_name
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            success = False
            continue

        try:
            is_unique = check_uniqueness(csv_path)
        except ValueError as exc:
            print(exc)
            success = False
        else:
            success = success and is_unique

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
