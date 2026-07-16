"""Project paths and Spotify tracks data download/load helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

# Project root: spotify-genre-prediction/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KAGGLE_DATASET = "maharshipandya/spotify-tracks-dataset"
KAGGLE_URL = "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
RAW_CSV_NAME = "dataset.csv"
RAW_CSV_PATH = DATA_RAW_DIR / RAW_CSV_NAME

# Audio features used later for genre prediction
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "duration_ms",
]

TARGET_COLUMN = "track_genre"


def ensure_data_dirs() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def raw_csv_exists() -> bool:
    return RAW_CSV_PATH.is_file()


def _unzip_archives_in_raw() -> None:
    """Unzip any .zip files sitting in data/raw/ (manual or API download)."""
    for zip_path in DATA_RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_RAW_DIR)
        print(f"Extracted {zip_path.name} -> {DATA_RAW_DIR}")


def _find_csv_after_download() -> Path | None:
    if RAW_CSV_PATH.is_file():
        return RAW_CSV_PATH
    matches = sorted(DATA_RAW_DIR.glob("*.csv"))
    return matches[0] if matches else None


def download_dataset(force: bool = False) -> Path:
    """
    Download the Kaggle Spotify Tracks dataset into data/raw/.

    Requires either:
      - ~/.kaggle/kaggle.json  (standard Kaggle API token), or
      - KAGGLE_USERNAME and KAGGLE_KEY environment variables

    Before the first API download, open the dataset page in a browser while
    logged into Kaggle and accept any dataset terms if prompted:
      https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

    Manual fallback (no API):
      1. Download the dataset ZIP from the Kaggle page above
      2. Place the ZIP or dataset.csv into data/raw/
      3. Re-run this function / script (it will unzip if needed)
    """
    ensure_data_dirs()

    if raw_csv_exists() and not force:
        print(f"Dataset already present: {RAW_CSV_PATH}")
        return RAW_CSV_PATH

    if force and raw_csv_exists():
        RAW_CSV_PATH.unlink()

    # Prefer API download; fall back to local zip/csv if API is unavailable
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {KAGGLE_DATASET} via Kaggle API...")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DATA_RAW_DIR),
            unzip=True,
            quiet=False,
        )
    except Exception as exc:  # noqa: BLE001 - show actionable guidance
        print("Kaggle API download failed.")
        print(f"  Reason: {exc}")
        print()
        print("Manual download steps:")
        print(f"  1. Open {KAGGLE_URL}")
        print("  2. Download the dataset ZIP")
        print(f"  3. Move the ZIP or {RAW_CSV_NAME} into: {DATA_RAW_DIR}")
        print("  4. Re-run this script")
        print()
        print("API credential setup (optional, for automatic download):")
        print("  1. Kaggle account -> Settings -> API -> Create New Token")
        print("  2. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
        print("  4. Accept dataset terms on the Kaggle dataset page")

    _unzip_archives_in_raw()
    found = _find_csv_after_download()
    if found is None:
        raise FileNotFoundError(
            f"Could not find {RAW_CSV_NAME} (or any CSV) in {DATA_RAW_DIR}. "
            "Download the dataset manually or fix Kaggle API credentials."
        )

    if found != RAW_CSV_PATH:
        found.rename(RAW_CSV_PATH)
        print(f"Renamed {found.name} -> {RAW_CSV_PATH.name}")

    print(f"Ready: {RAW_CSV_PATH}")
    return RAW_CSV_PATH


def load_raw_tracks(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the raw Spotify tracks CSV into a DataFrame."""
    path = Path(csv_path) if csv_path else RAW_CSV_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            "Run: python scripts/download_data.py\n"
            f"Or place {RAW_CSV_NAME} in {DATA_RAW_DIR}"
        )

    df = pd.read_csv(path, index_col=0) if _has_unnamed_index(path) else pd.read_csv(path)
    return df


def _has_unnamed_index(path: Path) -> bool:
    """Detect the leading unnamed index column common in this dataset."""
    header = pd.read_csv(path, nrows=0)
    first = header.columns[0]
    return str(first).startswith("Unnamed") or first == ""


def summarize_tracks(df: pd.DataFrame) -> dict:
    """Return a small summary dict useful for prints / notebooks / later app UI."""
    missing_features = [c for c in AUDIO_FEATURES if c not in df.columns]
    has_target = TARGET_COLUMN in df.columns
    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "n_genres": int(df[TARGET_COLUMN].nunique()) if has_target else None,
        "top_genres": (
            df[TARGET_COLUMN].value_counts().head(10).to_dict() if has_target else None
        ),
        "missing_audio_features": missing_features,
        "has_target": has_target,
    }
    return summary


def print_load_report(df: pd.DataFrame) -> None:
    summary = summarize_tracks(df)
    print(f"Shape: {summary['n_rows']:,} rows x {summary['n_cols']} columns")
    print(f"Columns ({len(summary['columns'])}): {summary['columns']}")
    if summary["has_target"]:
        print(f"Genres (track_genre): {summary['n_genres']} unique")
        print("Top 10 genres by track count:")
        for genre, count in summary["top_genres"].items():
            print(f"  {genre}: {count:,}")
    else:
        print(f"WARNING: expected target column '{TARGET_COLUMN}' not found.")
    if summary["missing_audio_features"]:
        print(f"WARNING: missing audio features: {summary['missing_audio_features']}")
    else:
        print("All expected audio feature columns are present.")
    print("\nFirst 5 rows:")
    print(df.head())
