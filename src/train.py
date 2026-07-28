"""Train and persist a supervised genre classifier from Spotify audio features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.clean import (
    DEFAULT_MIN_GENRE_COUNT,
    clean_tracks,
    print_cleaning_report,
)
from src.data_io import (
    AUDIO_FEATURES,
    DATA_PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_raw_tracks,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "genre_classifier.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select audio features + genre label; drop incomplete rows."""
    missing_cols = [c for c in AUDIO_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    subset = df[AUDIO_FEATURES + [TARGET_COLUMN]].dropna()
    X = subset[AUDIO_FEATURES]
    y = subset[TARGET_COLUMN].astype(str)
    return X, y


def build_pipeline(
    n_estimators: int = 100,
    max_depth: int = 18,
    min_samples_leaf: int = 5,
) -> Pipeline:
    """
    Standardize features, then classify with a depth-limited Random Forest.

    Depth / leaf limits keep the saved .joblib small enough for GitHub (<100MB)
    while remaining strong enough for the collapsed-genre task.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def train_genre_classifier(
    df: pd.DataFrame | None = None,
    n_estimators: int = 100,
    max_depth: int = 18,
    min_samples_leaf: int = 5,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_genre_count: int = DEFAULT_MIN_GENRE_COUNT,
    collapse_genres: bool = True,
    clean: bool = True,
) -> dict[str, Any]:
    """
    Clean/collapse genres, train on audio features, evaluate on a 20% holdout,
    and save model artifacts.
    """
    if df is None:
        df = load_raw_tracks()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaning_report: dict[str, Any] | None = None
    if clean:
        df, cleaning_report = clean_tracks(
            df=df,
            min_genre_count=min_genre_count,
            collapse_genres=collapse_genres,
            save=True,
        )
        print_cleaning_report(cleaning_report)

    X, y = prepare_xy(df)
    print(f"Rows used for train/test split: {len(X):,}")
    print(f"Features ({len(AUDIO_FEATURES)}): {AUDIO_FEATURES}")
    print(f"Unique genres: {y.nunique()}")
    print(f"Test holdout: {test_size:.0%} (stratified)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    print(
        "Fitting RandomForest pipeline "
        f"(n_estimators={n_estimators}, max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf})..."
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    clf = pipeline.named_steps["clf"]
    importances = {
        feature: float(importance)
        for feature, importance in zip(AUDIO_FEATURES, clf.feature_importances_)
    }
    importances_sorted = dict(
        sorted(importances.items(), key=lambda item: item[1], reverse=True)
    )

    metrics: dict[str, Any] = {
        "model_type": "Pipeline(StandardScaler, RandomForestClassifier)",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "test_size": test_size,
        "random_state": random_state,
        "cleaning": cleaning_report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(AUDIO_FEATURES),
        "n_classes": int(y.nunique()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "feature_importances": importances_sorted,
        "classification_report": report,
    }

    # compress=3 keeps the artifact GitHub-friendly (100MB file limit)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    metrics["model_size_mb"] = round(model_size_mb, 2)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    FEATURE_LIST_PATH.write_text(json.dumps(AUDIO_FEATURES, indent=2))

    sample_path = DATA_PROCESSED_DIR / "train_sample_head.csv"
    X_train.head(100).assign(**{TARGET_COLUMN: y_train.head(100).values}).to_csv(
        sample_path, index=False
    )

    print()
    print("=== Test-set results (held-out 20%) ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print()
    print("Top feature importances:")
    for name, value in list(importances_sorted.items())[:8]:
        print(f"  {name}: {value:.4f}")
    print()
    print(f"Saved model:   {MODEL_PATH} ({model_size_mb:.1f} MB)")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved features:{FEATURE_LIST_PATH}")
    if model_size_mb >= 95:
        print(
            "WARNING: model is near/over GitHub's 100MB limit. "
            "Retrain with fewer trees or a smaller max_depth."
        )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "model_path": MODEL_PATH,
        "metrics_path": METRICS_PATH,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "cleaning_report": cleaning_report,
    }


def load_trained_model(model_path: Path | None = None) -> Pipeline:
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"No saved model at {path}. Run: python scripts/train_model.py"
        )
    return joblib.load(path)


def load_metrics(metrics_path: Path | None = None) -> dict[str, Any]:
    path = Path(metrics_path) if metrics_path else METRICS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"No metrics file at {path}")
    return json.loads(path.read_text())
