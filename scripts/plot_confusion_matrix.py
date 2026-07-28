"""
Create visual confusion matrices for the held-out test set.

Usage (from project root, conda env activated):
  python scripts/plot_confusion_matrix.py

Outputs:
  outputs/confusion_matrix.png
  outputs/confusion_matrix_normalized.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.plots import make_confusion_matrix_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and save confusion matrices for the saved genre model."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Optional path to test_predictions.csv (default: outputs/test_predictions.csv).",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions) if args.predictions else None
    result = make_confusion_matrix_figures(predictions_path=pred_path, show=False)

    print()
    print("Confusion matrices created.")
    print(f"Test rows: {result['n_test']:,}")
    print(f"Genres:    {len(result['labels'])}")
    print(f"Counts:    {result['raw_path']}")
    print(f"Normalized:{result['normalized_path']}")


if __name__ == "__main__":
    main()
