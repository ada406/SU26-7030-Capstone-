"""Evaluate the saved model on the held-out test set and write predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.clean import DEFAULT_MIN_GENRE_COUNT, CLEANED_CSV_PATH, clean_tracks
from src.data_io import AUDIO_FEATURES, PROJECT_ROOT, TARGET_COLUMN, load_raw_tracks
from src.train import (
    RANDOM_STATE,
    TEST_SIZE,
    load_metrics,
    load_trained_model,
    prepare_xy,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"


def load_modeling_frame(
    min_genre_count: int = DEFAULT_MIN_GENRE_COUNT,
    collapse_genres: bool = True,
    prefer_saved_clean: bool = True,
) -> pd.DataFrame:
    """Load cleaned data (from disk if present) for evaluation."""
    if prefer_saved_clean and CLEANED_CSV_PATH.is_file():
        print(f"Loading cleaned data: {CLEANED_CSV_PATH}")
        return pd.read_csv(CLEANED_CSV_PATH)

    print("Cleaned CSV not found; running cleaning pipeline...")
    df, _ = clean_tracks(
        df=load_raw_tracks(),
        min_genre_count=min_genre_count,
        collapse_genres=collapse_genres,
        save=True,
    )
    return df


def get_held_out_test_split(
    df: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_genre_count: int = DEFAULT_MIN_GENRE_COUNT,
    collapse_genres: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Rebuild the same stratified 80/20 split used during training.

    Uses random_state=42 and test_size=0.2 by default so evaluation matches
    models/metrics.json when the same cleaning settings were used.
    """
    if df is None:
        df = load_modeling_frame(
            min_genre_count=min_genre_count,
            collapse_genres=collapse_genres,
        )

    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, y_train, X_test, y_test


def evaluate_saved_model(
    n_examples: int = 15,
    save_predictions: bool = True,
) -> dict[str, Any]:
    """
    Load models/genre_classifier.joblib, score the held-out test set,
    print sample predictions, and optionally save all test predictions.
    """
    model = load_trained_model()
    saved = load_metrics()

    _, _, X_test, y_test = get_held_out_test_split()
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    confidence = proba.max(axis=1)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    print("=== Evaluation of saved model on held-out test set (20%) ===")
    print(f"Test rows:     {len(y_test):,}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    print(f"Saved metrics accuracy (from training): {saved.get('accuracy')}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    results = X_test.copy()
    results["true_genre"] = y_test.to_numpy()
    results["predicted_genre"] = y_pred
    results["confidence"] = confidence
    results["correct"] = results["true_genre"] == results["predicted_genre"]

    print(f"=== Sample predictions (first {n_examples}) ===")
    sample_cols = ["true_genre", "predicted_genre", "confidence", "correct"]
    print(results[sample_cols].head(n_examples).to_string(index=False))
    print()
    print(
        f"Correct on full test set: "
        f"{int(results['correct'].sum()):,} / {len(results):,} "
        f"({results['correct'].mean():.1%})"
    )

    if save_predictions:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Saved all test predictions: {PREDICTIONS_PATH}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_test": int(len(y_test)),
        "predictions": results,
        "predictions_path": PREDICTIONS_PATH if save_predictions else None,
    }


def predict_genres(features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict genres for a DataFrame of audio features.

    `features` must include the columns in AUDIO_FEATURES.
    """
    missing = [c for c in AUDIO_FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"Missing audio feature columns: {missing}")

    model = load_trained_model()
    X = features[AUDIO_FEATURES]
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    # top-3 labels for each row
    top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
    classes = np.asarray(model.classes_)

    out = features.copy()
    out["predicted_genre"] = y_pred
    out["confidence"] = conf
    out["top3_genres"] = [
        ", ".join(classes[idx_row]) for idx_row in top3_idx
    ]
    return out
