"""Check CSV files for uniqueness of the `id` column.

This script inspects the `train.csv` and `test.csv` files under the
project's `data` directory and verifies that the values in their `id`
columns are unique.  It prints a short report for each file indicating
whether duplicates were found and, when duplicates exist, whether the
remaining columns also match.  Rows that are identical across every
column are treated as a single entry during the check.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence


def read_rows(csv_path: Path, required_column: str) -> tuple[List[dict[str, str]], Sequence[str]]:
    """Read ``csv_path`` and return its rows and field names.

    The ``required_column`` must be present in the CSV.
    """

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or required_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{required_column}' not found in {csv_path.name}."
            )
        rows = list(reader)
        return rows, reader.fieldnames


def collapse_identical_rows(
    rows: List[dict[str, str]], fieldnames: Sequence[str]
) -> tuple[List[dict[str, str]], int]:
    """Return ``rows`` without exact duplicates along with removal count."""

    seen_rows: set[tuple[str, ...]] = set()
    unique_rows: List[dict[str, str]] = []
    duplicates_removed = 0

    for row in rows:
        key = tuple(row[field] for field in fieldnames)
        if key in seen_rows:
            duplicates_removed += 1
            continue
        seen_rows.add(key)
        unique_rows.append(row)

    return unique_rows, duplicates_removed


def check_uniqueness(csv_path: Path, column_name: str = "id") -> bool:
    """Check that ``column_name`` values in ``csv_path`` are unique.

    Returns ``True`` when all values are unique, otherwise ``False``.
    """

    rows, fieldnames = read_rows(csv_path, column_name)

    rows, removed_duplicates = collapse_identical_rows(rows, fieldnames)
    if removed_duplicates:
        print(
            f"{csv_path.name}: Removed {removed_duplicates} completely identical "
            "duplicate row(s)."
        )

    values = [row[column_name] for row in rows]
    counter = Counter(values)
    duplicates = {value: count for value, count in counter.items() if count > 1}

    if duplicates:
        remaining_fields = [field for field in fieldnames if field != column_name]
        print(f"{csv_path.name}: Found duplicates in '{column_name}' column.")
        for value, count in duplicates.items():
            print(f"  Value '{value}' appears {count} times")

            duplicate_rows = [
                tuple(row[field] for field in remaining_fields)
                for row in rows
                if row[column_name] == value
            ]
            unique_remaining = set(duplicate_rows)
            if not remaining_fields:
                print("    No additional columns to compare.")
            elif len(unique_remaining) == 1:
                print("    Remaining columns are identical across duplicates.")
            else:
                print("    Remaining columns differ across duplicates:")
                for idx, values_tuple in enumerate(duplicate_rows, start=1):
                    formatted_values = ", ".join(
                        f"{field}='{field_value}'"
                        for field, field_value in zip(remaining_fields, values_tuple)
                    )
                    print(f"      Duplicate #{idx}: {formatted_values}")
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
