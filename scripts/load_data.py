"""
Load the Spotify Tracks dataset and print a short validation report.

Usage (from project root, with conda env activated):
  python scripts/load_data.py
  python scripts/load_data.py --download   # download first if missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import (  # noqa: E402
    download_dataset,
    load_raw_tracks,
    print_load_report,
    raw_csv_exists,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load data/raw/dataset.csv and print shape, genres, and columns."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset first if it is missing.",
    )
    args = parser.parse_args()

    if args.download or not raw_csv_exists():
        if not raw_csv_exists():
            print("Raw CSV not found; attempting download...")
        download_dataset()

    df = load_raw_tracks()
    print_load_report(df)


if __name__ == "__main__":
    main()
