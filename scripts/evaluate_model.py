"""
Evaluate the saved genre classifier on the held-out 20% test set
and write predictions to outputs/test_predictions.csv.

Usage (from project root, env activated, data available):
  python scripts/evaluate_model.py
  python scripts/evaluate_model.py --examples 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_saved_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load models/genre_classifier.joblib, score the held-out test set, "
            "print sample predictions, and save outputs/test_predictions.csv."
        )
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=15,
        help="How many sample prediction rows to print (default: 15).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write outputs/test_predictions.csv.",
    )
    args = parser.parse_args()

    result = evaluate_saved_model(
        n_examples=args.examples,
        save_predictions=not args.no_save,
    )
    print()
    print("Evaluation complete.")
    print(f"Test accuracy: {result['accuracy']:.4f}")
    if result["predictions_path"]:
        print(f"Predictions file: {result['predictions_path']}")


if __name__ == "__main__":
    main()
