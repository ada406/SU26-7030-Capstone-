"""
Download the Spotify Tracks dataset into data/raw/.

Usage (from project root, with conda env activated):
  python scripts/download_data.py
  python scripts/download_data.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/download_data.py` without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import download_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle Spotify Tracks dataset into data/raw/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if dataset.csv already exists.",
    )
    args = parser.parse_args()
    path = download_dataset(force=args.force)
    print(f"Done. CSV path: {path}")


if __name__ == "__main__":
    main()
