"""
Train a supervised genre classifier from Spotify audio features and save it.

Pipeline:
  1) Clean data
  2) Collapse similar fine-grained genres into parent genres
  3) Deduplicate tracks (prefer common parent genres)
  4) Drop rare parent genres
  5) Hold out 20% for testing
  6) Tune/train RandomForest and/or HistGradientBoosting, save the best model

Usage (from project root, conda env activated, dataset in data/raw/):
  python scripts/train_model.py
  python scripts/train_model.py --model compare --tune
  python scripts/train_model.py --model hgb --tune --tune-iter 12
  python scripts/train_model.py --model rf --no-tune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clean import DEFAULT_MIN_GENRE_COUNT  # noqa: E402
from src.train import SUPPORTED_MODELS, train_genre_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean + collapse Spotify genres, train/tune classifiers on audio "
            "features, evaluate on a 20% holdout, and save the best model."
        )
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="compare",
        help=(
            "Which model(s) to train: rf, hgb, or compare (default: compare). "
            "compare trains both and keeps the better test macro-F1 model."
        ),
    )
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run RandomizedSearchCV hyperparameter tuning (default: true).",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=10,
        help="RandomizedSearchCV iterations per model (default: 10).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="CV folds used during tuning (default: 3).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="RF trees when --no-tune (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="RF max depth when --no-tune (default: 20).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="RF min samples per leaf when --no-tune (default: 2).",
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
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        test_size=args.test_size,
        min_genre_count=args.min_genre_count,
        collapse_genres=not args.no_collapse,
        clean=not args.no_clean,
        model=args.model,
        tune=args.tune,
        tune_iter=args.tune_iter,
        cv_folds=args.cv,
    )
    print()
    print("Training complete.")
    print(f"Selected model: {result['selected_model']}")
    print(f"Model file: {result['model_path']}")
    print(f"Test accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Test macro F1: {result['metrics']['macro_f1']:.4f}")
    print(f"Genres used: {result['metrics']['n_classes']}")


if __name__ == "__main__":
    main()
