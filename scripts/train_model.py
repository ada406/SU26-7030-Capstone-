"""
Train a supervised genre classifier from Spotify audio features and save it.

Pipeline:
  1) Clean data
  2) Collapse similar fine-grained genres into parent genres
  3) Deduplicate tracks (prefer common parent genres)
  4) Drop rare parent genres
  5) Hold out 20% for testing
  6) Train RandomForest and save model/metrics

Usage (from project root, conda env activated, dataset in data/raw/):
  python scripts/train_model.py
  python scripts/train_model.py --min-genre-count 500
  python scripts/train_model.py --no-collapse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clean import DEFAULT_MIN_GENRE_COUNT  # noqa: E402
from src.train import train_genre_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean + collapse Spotify genres, train a Random Forest on audio "
            "features, evaluate on a 20% holdout, and save the model."
        )
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest (default: 200).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing (default: 0.2).",
    )
    parser.add_argument(
        "--min-genre-count",
        type=int,
        default=DEFAULT_MIN_GENRE_COUNT,
        help=(
            "Drop parent genres with fewer than this many tracks after cleaning "
            f"(default: {DEFAULT_MIN_GENRE_COUNT})."
        ),
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="Do not collapse similar genres into parent labels.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning entirely (not recommended; comparison only).",
    )
    args = parser.parse_args()

    result = train_genre_classifier(
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        min_genre_count=args.min_genre_count,
        collapse_genres=not args.no_collapse,
        clean=not args.no_clean,
    )
    print()
    print("Training complete.")
    print(f"Model file: {result['model_path']}")
    print(f"Test accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Genres used: {result['metrics']['n_classes']}")


if __name__ == "__main__":
    main()
