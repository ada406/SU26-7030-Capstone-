"""Plotting helpers for model evaluation figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data_io import PROJECT_ROOT
from src.evaluate import PREDICTIONS_PATH, evaluate_saved_model

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_NORM_PATH = OUTPUTS_DIR / "confusion_matrix_normalized.png"


def load_prediction_labels(
    predictions_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load true/predicted labels for the test set.

    Prefers outputs/test_predictions.csv when present; otherwise runs evaluation.
    """
    path = Path(predictions_path) if predictions_path else PREDICTIONS_PATH
    if path.is_file():
        print(f"Loading predictions from {path}")
        df = pd.read_csv(path)
        y_true = df["true_genre"].astype(str).to_numpy()
        y_pred = df["predicted_genre"].astype(str).to_numpy()
    else:
        print("Predictions CSV not found; evaluating saved model...")
        result = evaluate_saved_model(n_examples=0, save_predictions=True)
        df = result["predictions"]
        y_true = df["true_genre"].astype(str).to_numpy()
        y_pred = df["predicted_genre"].astype(str).to_numpy()

    # Stable alphabetical order (no need to load the .joblib just for labels)
    labels = sorted(set(y_true) | set(y_pred))
    return y_true, y_pred, labels


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    *,
    normalize: bool = False,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> Path:
    """
    Create and save a confusion-matrix heatmap.

    normalize=False -> raw counts
    normalize=True  -> row-normalized (share of each true genre)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = np.divide(cm, row_sums, where=row_sums != 0)
        cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
        cbar_label = "Row share"
        default_title = "Normalized confusion matrix (test set)"
        default_path = CONFUSION_MATRIX_NORM_PATH
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"
        default_title = "Confusion matrix (test set)"
        default_path = CONFUSION_MATRIX_PATH

    out_path = Path(save_path) if save_path else default_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(labels)
    fig_w = max(10, 0.55 * n + 4)
    fig_h = max(8, 0.55 * n + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": cbar_label},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    ax.set_title(title or default_title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


def make_confusion_matrix_figures(
    predictions_path: Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Build both raw-count and normalized confusion matrix PNGs."""
    y_true, y_pred, labels = load_prediction_labels(predictions_path)
    raw_path = plot_confusion_matrix(
        y_true, y_pred, labels, normalize=False, show=show
    )
    norm_path = plot_confusion_matrix(
        y_true, y_pred, labels, normalize=True, show=show
    )
    return {
        "labels": labels,
        "n_test": int(len(y_true)),
        "raw_path": raw_path,
        "normalized_path": norm_path,
    }
