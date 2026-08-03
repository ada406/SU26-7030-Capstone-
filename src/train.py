"""Train and persist a supervised genre classifier from Spotify audio features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

SUPPORTED_MODELS = ("rf", "hgb", "compare")


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select audio features + genre label; drop incomplete rows."""
    missing_cols = [c for c in AUDIO_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    subset = df[AUDIO_FEATURES + [TARGET_COLUMN]].dropna()
    X = subset[AUDIO_FEATURES]
    y = subset[TARGET_COLUMN].astype(str)
    return X, y


def build_rf_pipeline(
    n_estimators: int = 100,
    max_depth: int | None = 20,
    min_samples_leaf: int = 2,
    max_features: str | float | None = 0.5,
) -> Pipeline:
    """
    Standardize features, then classify with a Random Forest.

    Depth / leaf limits help keep the saved .joblib GitHub-friendly (<100MB).
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
                    max_features=max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def build_pipeline(
    n_estimators: int = 100,
    max_depth: int = 20,
    min_samples_leaf: int = 2,
) -> Pipeline:
    """Backward-compatible alias for the default Random Forest pipeline."""
    return build_rf_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=0.5,
    )


def build_hgb_pipeline(
    learning_rate: float = 0.1,
    max_depth: int | None = 12,
    max_iter: int = 200,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
) -> Pipeline:
    """Standardize features, then classify with HistGradientBoosting."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    max_iter=max_iter,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def rf_param_distributions() -> dict[str, list[Any]]:
    return {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [14, 18, 22, 28],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features": ["sqrt", 0.5],
    }


def hgb_param_distributions() -> dict[str, list[Any]]:
    return {
        "clf__learning_rate": [0.05, 0.1, 0.15],
        "clf__max_depth": [6, 10, 15, None],
        "clf__max_iter": [150, 250, 350],
        "clf__min_samples_leaf": [10, 20, 40],
        "clf__l2_regularization": [0.0, 0.1, 1.0],
    }


def _score_split(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    y_pred = pipeline.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }


def _feature_importances(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
) -> dict[str, float]:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = {
            feature: float(importance)
            for feature, importance in zip(AUDIO_FEATURES, clf.feature_importances_)
        }
    else:
        # HistGradientBoosting has no impurity importances; use a small permutation sample.
        result = permutation_importance(
            pipeline,
            X_sample,
            y_sample,
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="f1_macro",
        )
        importances = {
            feature: float(importance)
            for feature, importance in zip(AUDIO_FEATURES, result.importances_mean)
        }
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def fit_model_candidate(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    tune: bool = False,
    tune_iter: int = 10,
    cv_folds: int = 3,
    n_estimators: int = 100,
    max_depth: int = 20,
    min_samples_leaf: int = 2,
) -> dict[str, Any]:
    """Fit one model (optionally with RandomizedSearchCV) and return artifacts."""
    if model_name == "rf":
        pipeline = build_rf_pipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        param_distributions = rf_param_distributions()
        default_label = (
            f"RandomForest(n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_leaf={min_samples_leaf})"
        )
    elif model_name == "hgb":
        pipeline = build_hgb_pipeline()
        param_distributions = hgb_param_distributions()
        default_label = "HistGradientBoosting(defaults)"
    else:
        raise ValueError(f"Unknown model_name={model_name!r}; use 'rf' or 'hgb'.")

    best_params: dict[str, Any] | None = None
    cv_macro_f1: float | None = None

    if tune:
        print(
            f"Tuning {model_name} with RandomizedSearchCV "
            f"(n_iter={tune_iter}, cv={cv_folds}, scoring=f1_macro)..."
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=tune_iter,
            scoring="f1_macro",
            cv=StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
            ),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = {
            key.replace("clf__", ""): value for key, value in search.best_params_.items()
        }
        cv_macro_f1 = float(search.best_score_)
        print(f"  Best CV macro F1: {cv_macro_f1:.4f}")
        print(f"  Best params: {best_params}")
        model_label = f"tuned {model_name}: {best_params}"
    else:
        print(f"Fitting {default_label}...")
        pipeline.fit(X_train, y_train)
        model_label = default_label

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "best_params": best_params,
        "cv_macro_f1": cv_macro_f1,
        "model_label": model_label,
        "tuned": tune,
    }


def train_genre_classifier(
    df: pd.DataFrame | None = None,
    n_estimators: int = 100,
    max_depth: int = 20,
    min_samples_leaf: int = 2,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_genre_count: int = DEFAULT_MIN_GENRE_COUNT,
    collapse_genres: bool = True,
    clean: bool = True,
    model: str = "compare",
    tune: bool = True,
    tune_iter: int = 10,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """
    Clean/collapse genres, train one or more classifiers, evaluate on a 20% holdout,
    and save the best model artifacts.

    model:
      - "rf": Random Forest only
      - "hgb": HistGradientBoosting only
      - "compare": tune/train both and keep the better test macro-F1 model
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {SUPPORTED_MODELS}, got {model!r}")

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
    print(f"Model mode: {model} | tune={tune}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    candidates = ["rf", "hgb"] if model == "compare" else [model]
    comparison_rows: list[dict[str, Any]] = []
    fitted: list[dict[str, Any]] = []

    for name in candidates:
        candidate = fit_model_candidate(
            name,
            X_train,
            y_train,
            tune=tune,
            tune_iter=tune_iter,
            cv_folds=cv_folds,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        test_scores = _score_split(candidate["pipeline"], X_test, y_test)
        row = {
            "model_name": name,
            "tuned": tune,
            "cv_macro_f1": candidate["cv_macro_f1"],
            "best_params": candidate["best_params"],
            **test_scores,
        }
        comparison_rows.append(row)
        candidate["test_scores"] = test_scores
        fitted.append(candidate)
        print(
            f"  {name} test accuracy={test_scores['accuracy']:.4f} "
            f"macro_f1={test_scores['macro_f1']:.4f}"
        )

    # Prefer higher test macro F1; break ties with accuracy.
    winner = max(
        fitted,
        key=lambda c: (c["test_scores"]["macro_f1"], c["test_scores"]["accuracy"]),
    )
    pipeline = winner["pipeline"]
    y_pred = pipeline.predict(X_test)
    accuracy = winner["test_scores"]["accuracy"]
    macro_f1 = winner["test_scores"]["macro_f1"]
    weighted_f1 = winner["test_scores"]["weighted_f1"]
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Keep permutation importance cheap on a stratified sample for HGB.
    sample_n = min(4000, len(X_test))
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(len(X_test), size=sample_n, replace=False)
    importances_sorted = _feature_importances(
        pipeline, X_test.iloc[sample_idx], y_test.iloc[sample_idx]
    )

    clf = pipeline.named_steps["clf"]
    clf_params = {
        key: value
        for key, value in clf.get_params().items()
        if key
        in {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "learning_rate",
            "max_iter",
            "l2_regularization",
        }
    }

    metrics: dict[str, Any] = {
        "model_type": f"Pipeline(StandardScaler, {type(clf).__name__})",
        "selected_model": winner["model_name"],
        "tuned": tune,
        "best_params": winner["best_params"] or clf_params,
        "cv_macro_f1": winner["cv_macro_f1"],
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
        "model_comparison": comparison_rows,
        # Keep legacy keys populated when RF wins (handy for older notebooks).
        "n_estimators": clf_params.get("n_estimators"),
        "max_depth": clf_params.get("max_depth"),
        "min_samples_leaf": clf_params.get("min_samples_leaf"),
    }

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    metrics["model_size_mb"] = round(model_size_mb, 2)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    FEATURE_LIST_PATH.write_text(json.dumps(AUDIO_FEATURES, indent=2))
    COMPARISON_PATH.write_text(json.dumps(comparison_rows, indent=2))

    sample_path = DATA_PROCESSED_DIR / "train_sample_head.csv"
    X_train.head(100).assign(**{TARGET_COLUMN: y_train.head(100).values}).to_csv(
        sample_path, index=False
    )

    print()
    print("=== Model comparison (held-out 20% test) ===")
    for row in comparison_rows:
        marker = " <-- selected" if row["model_name"] == winner["model_name"] else ""
        print(
            f"  {row['model_name']}: acc={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} weighted_f1={row['weighted_f1']:.4f}"
            f"{marker}"
        )
    print()
    print("=== Selected model test-set results ===")
    print(f"Model:       {metrics['model_type']}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    if winner["best_params"]:
        print(f"Best params: {winner['best_params']}")
    print()
    print("Top feature importances:")
    for name, value in list(importances_sorted.items())[:8]:
        print(f"  {name}: {value:.4f}")
    print()
    print(f"Saved model:      {MODEL_PATH} ({model_size_mb:.1f} MB)")
    print(f"Saved metrics:    {METRICS_PATH}")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print(f"Saved features:   {FEATURE_LIST_PATH}")
    if model_size_mb >= 95:
        print(
            "WARNING: model is near/over GitHub's 100MB limit. "
            "Prefer HistGradientBoosting or a shallower/smaller forest."
        )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "model_path": MODEL_PATH,
        "metrics_path": METRICS_PATH,
        "comparison_path": COMPARISON_PATH,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "cleaning_report": cleaning_report,
        "comparison": comparison_rows,
        "selected_model": winner["model_name"],
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
