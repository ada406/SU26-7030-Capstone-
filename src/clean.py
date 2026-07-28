"""Clean the Spotify tracks dataset before model training."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_io import (
    AUDIO_FEATURES,
    DATA_PROCESSED_DIR,
    TARGET_COLUMN,
    load_raw_tracks,
)
from src.genre_map import collapse_genres_series

# After collapse + dedupe, keep parent genres with enough examples.
DEFAULT_MIN_GENRE_COUNT = 500

CLEANED_CSV_PATH = DATA_PROCESSED_DIR / "tracks_clean.csv"


def _dedupe_tracks_preferring_common_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per track, preferring globally common genre labels.

    Example: if the same track_id is labeled both "pop" and "folk",
    and "pop" is more frequent overall, keep "pop".
    """
    if "track_id" not in df.columns:
        dedupe_cols = list(AUDIO_FEATURES)
        for optional in ("track_name", "artists"):
            if optional in df.columns:
                dedupe_cols.append(optional)
        return df.drop_duplicates(subset=dedupe_cols, keep="first")

    genre_freq = df[TARGET_COLUMN].value_counts()
    ranked = df.copy()
    ranked["_genre_freq"] = ranked[TARGET_COLUMN].map(genre_freq).fillna(0).astype(int)
    ranked = ranked.sort_values(
        by=["track_id", "_genre_freq"],
        ascending=[True, False],
        kind="mergesort",
    )
    ranked = ranked.drop_duplicates(subset=["track_id"], keep="first")
    return ranked.drop(columns=["_genre_freq"]).reset_index(drop=True)


def clean_tracks(
    df: pd.DataFrame | None = None,
    min_genre_count: int = DEFAULT_MIN_GENRE_COUNT,
    collapse_genres: bool = True,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean raw Spotify tracks for supervised genre classification.

    Steps:
      1. Drop rows missing audio features / genre
      2. Normalize genre labels
      3. Collapse similar fine-grained genres into parent genres
      4. Drop exact duplicate rows
      5. Keep one row per track, preferring common parent genres
      6. Drop rare parent genres below min_genre_count
      7. Optionally save data/processed/tracks_clean.csv
    """
    if df is None:
        df = load_raw_tracks()

    raw_rows = int(len(df))
    raw_genres = int(df[TARGET_COLUMN].nunique()) if TARGET_COLUMN in df.columns else None

    required = AUDIO_FEATURES + [TARGET_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    cleaned = df.copy()

    # 1) Drop incomplete feature/label rows
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=required)
    dropped_na = before - len(cleaned)

    # 2) Normalize genre strings
    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN].astype(str).str.strip().str.lower()
    )
    cleaned = cleaned[cleaned[TARGET_COLUMN].str.len() > 0]
    genres_before_collapse = int(cleaned[TARGET_COLUMN].nunique())

    # 3) Collapse similar genres into broader parents
    if collapse_genres:
        original_genres = cleaned[TARGET_COLUMN].copy()
        cleaned[TARGET_COLUMN] = collapse_genres_series(cleaned[TARGET_COLUMN])
        n_relabeled = int((original_genres != cleaned[TARGET_COLUMN]).sum())
    else:
        n_relabeled = 0
    genres_after_collapse = int(cleaned[TARGET_COLUMN].nunique())

    # 4) Exact duplicate rows
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped_exact_dupes = before - len(cleaned)

    # 5) One row per track, preferring common parent genres
    before = len(cleaned)
    cleaned = _dedupe_tracks_preferring_common_genres(cleaned)
    dropped_track_id_dupes = before - len(cleaned)

    # 6) Drop rare parent genres
    genre_counts = cleaned[TARGET_COLUMN].value_counts()
    keep_genres = genre_counts[genre_counts >= min_genre_count].index
    dropped_genres = sorted(set(genre_counts.index) - set(keep_genres))
    before = len(cleaned)
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin(keep_genres)].copy()
    dropped_rare_rows = before - len(cleaned)

    if cleaned.empty or cleaned[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Cleaning removed too much data (need at least 2 genres with enough rows). "
            f"Try a lower --min-genre-count (current={min_genre_count})."
        )

    cleaned = cleaned.reset_index(drop=True)

    common_examples = [
        "pop",
        "rock",
        "metal",
        "hip-hop",
        "electronic",
        "jazz",
        "classical",
        "country",
        "latin",
        "r&b",
        "folk",
        "reggae",
        "blues",
        "ambient",
        "punk",
        "world",
        "kids",
    ]
    present = set(cleaned[TARGET_COLUMN])
    common_still_present = {
        g: int((cleaned[TARGET_COLUMN] == g).sum())
        for g in common_examples
        if g in present
    }
    common_missing_after_clean = [g for g in common_examples if g not in present]

    report: dict[str, Any] = {
        "raw_rows": raw_rows,
        "raw_genres": raw_genres,
        "dropped_na_rows": int(dropped_na),
        "collapse_genres": bool(collapse_genres),
        "genres_before_collapse": genres_before_collapse,
        "genres_after_collapse": genres_after_collapse,
        "rows_relabeled_by_collapse": n_relabeled,
        "dropped_exact_duplicate_rows": int(dropped_exact_dupes),
        "dropped_duplicate_track_rows": int(dropped_track_id_dupes),
        "dedupe_strategy": "prefer_common_genre_per_track_id",
        "min_genre_count": int(min_genre_count),
        "dropped_rare_genres": dropped_genres,
        "dropped_rare_genre_rows": int(dropped_rare_rows),
        "clean_rows": int(len(cleaned)),
        "clean_genres": int(cleaned[TARGET_COLUMN].nunique()),
        "common_genre_counts": common_still_present,
        "common_genres_missing": common_missing_after_clean,
        "genre_counts": cleaned[TARGET_COLUMN].value_counts().to_dict(),
    }

    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(CLEANED_CSV_PATH, index=False)
        report["cleaned_csv"] = str(CLEANED_CSV_PATH)

    return cleaned, report


def print_cleaning_report(report: dict[str, Any]) -> None:
    print("=== Cleaning report ===")
    print(f"Raw rows:                         {report['raw_rows']:,}")
    print(f"Raw genres:                       {report['raw_genres']}")
    print(f"Dropped NA rows:                  {report['dropped_na_rows']:,}")
    print(f"Collapse similar genres:          {report.get('collapse_genres')}")
    print(
        f"Genres before -> after collapse:  "
        f"{report.get('genres_before_collapse')} -> {report.get('genres_after_collapse')}"
    )
    print(f"Rows relabeled by collapse:       {report.get('rows_relabeled_by_collapse', 0):,}")
    print(f"Dropped exact duplicate rows:     {report['dropped_exact_duplicate_rows']:,}")
    print(f"Dropped duplicate-track rows:     {report['dropped_duplicate_track_rows']:,}")
    print(f"Dedupe strategy:                  {report.get('dedupe_strategy', 'n/a')}")
    print(f"Min genre count threshold:        {report['min_genre_count']}")
    print(
        f"Dropped rare genres ({len(report['dropped_rare_genres'])}): "
        f"{report['dropped_rare_genres'][:20]}"
        f"{' ...' if len(report['dropped_rare_genres']) > 20 else ''}"
    )
    print(f"Dropped rare-genre rows:          {report['dropped_rare_genre_rows']:,}")
    print(f"Clean rows:                       {report['clean_rows']:,}")
    print(f"Clean genres:                     {report['clean_genres']}")
    if report.get("common_genre_counts"):
        print(f"Parent genres kept (counts):      {report['common_genre_counts']}")
    if report.get("common_genres_missing"):
        print(f"Parent genres missing:            {report['common_genres_missing']}")
    if report.get("cleaned_csv"):
        print(f"Saved cleaned CSV:                {report['cleaned_csv']}")
    print()
