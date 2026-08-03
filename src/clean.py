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
from src.genre_map import (
    AMBIGUOUS_FINE_GENRES,
    PARENT_GENRES,
    collapse_genres_series,
    is_ambiguous_fine_genre,
)

# After collapse + dedupe, keep parent genres with enough examples.
DEFAULT_MIN_GENRE_COUNT = 500

CLEANED_CSV_PATH = DATA_PROCESSED_DIR / "tracks_clean.csv"


def _dedupe_tracks_drop_conflicting_parents(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Keep one row per track only when all of that track's labels agree on one parent.

    Tracks tagged with multiple conflicting parent genres after collapse are dropped
    entirely — those labels are too noisy for supervised training.
    """
    if "track_id" not in df.columns:
        dedupe_cols = list(AUDIO_FEATURES)
        for optional in ("track_name", "artists"):
            if optional in df.columns:
                dedupe_cols.append(optional)
        before = len(df)
        out = df.drop_duplicates(subset=dedupe_cols, keep="first")
        return out.reset_index(drop=True), int(before - len(out))

    parent_nunique = df.groupby("track_id")[TARGET_COLUMN].nunique()
    ambiguous_ids = parent_nunique[parent_nunique > 1].index
    n_ambiguous_tracks = int(len(ambiguous_ids))

    consistent = df[~df["track_id"].isin(ambiguous_ids)].copy()
    consistent = consistent.drop_duplicates(subset=["track_id"], keep="first")
    return consistent.reset_index(drop=True), n_ambiguous_tracks


def clean_tracks(
    df: pd.DataFrame | None = None,
    min_genre_count: int = DEFAULT_MIN_GENRE_COUNT,
    collapse_genres: bool = True,
    drop_ambiguous_fine_genres: bool = True,
    drop_conflicting_parents: bool = True,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean raw Spotify tracks for supervised genre classification.

    Steps:
      1. Drop rows missing audio features / genre
      2. Normalize genre labels
      3. Drop ambiguous fine-grained tags (mood/entertainment/vague)
      4. Collapse similar fine-grained genres into parent genres
      5. Drop exact duplicate rows
      6. Keep one row per track; drop tracks with conflicting parent labels
      7. Drop rare parent genres below min_genre_count
      8. Optionally save data/processed/tracks_clean.csv
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

    # 3) Drop ambiguous fine-grained labels before collapse
    dropped_ambiguous_fine_rows = 0
    dropped_ambiguous_fine_labels: list[str] = []
    if drop_ambiguous_fine_genres:
        ambiguous_mask = cleaned[TARGET_COLUMN].map(is_ambiguous_fine_genre)
        dropped_ambiguous_fine_labels = sorted(
            cleaned.loc[ambiguous_mask, TARGET_COLUMN].unique().tolist()
        )
        dropped_ambiguous_fine_rows = int(ambiguous_mask.sum())
        cleaned = cleaned.loc[~ambiguous_mask].copy()

    # 4) Collapse similar genres into broader parents
    if collapse_genres:
        original_genres = cleaned[TARGET_COLUMN].copy()
        cleaned[TARGET_COLUMN] = collapse_genres_series(cleaned[TARGET_COLUMN])
        n_relabeled = int((original_genres != cleaned[TARGET_COLUMN]).sum())
    else:
        n_relabeled = 0
    genres_after_collapse = int(cleaned[TARGET_COLUMN].nunique())

    # 5) Exact duplicate rows
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped_exact_dupes = before - len(cleaned)

    # 6) One row per track; optionally drop conflicting multi-parent tracks
    before = len(cleaned)
    if drop_conflicting_parents:
        cleaned, n_ambiguous_tracks = _dedupe_tracks_drop_conflicting_parents(cleaned)
        dropped_track_id_dupes = before - len(cleaned)
        dedupe_strategy = "drop_tracks_with_conflicting_parents"
    else:
        # Legacy fallback: prefer the globally more common parent label.
        genre_freq = cleaned[TARGET_COLUMN].value_counts()
        ranked = cleaned.copy()
        ranked["_genre_freq"] = ranked[TARGET_COLUMN].map(genre_freq).fillna(0).astype(int)
        ranked = ranked.sort_values(
            by=["track_id", "_genre_freq"],
            ascending=[True, False],
            kind="mergesort",
        )
        if "track_id" in ranked.columns:
            ranked = ranked.drop_duplicates(subset=["track_id"], keep="first")
        cleaned = ranked.drop(columns=["_genre_freq"], errors="ignore").reset_index(drop=True)
        dropped_track_id_dupes = before - len(cleaned)
        n_ambiguous_tracks = 0
        dedupe_strategy = "prefer_common_genre_per_track_id"

    # 7) Drop rare parent genres
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

    present = set(cleaned[TARGET_COLUMN])
    common_still_present = {
        g: int((cleaned[TARGET_COLUMN] == g).sum())
        for g in PARENT_GENRES
        if g in present
    }
    common_missing_after_clean = [g for g in PARENT_GENRES if g not in present]

    report: dict[str, Any] = {
        "raw_rows": raw_rows,
        "raw_genres": raw_genres,
        "dropped_na_rows": int(dropped_na),
        "collapse_genres": bool(collapse_genres),
        "drop_ambiguous_fine_genres": bool(drop_ambiguous_fine_genres),
        "ambiguous_fine_genre_list": sorted(AMBIGUOUS_FINE_GENRES),
        "dropped_ambiguous_fine_labels": dropped_ambiguous_fine_labels,
        "dropped_ambiguous_fine_rows": int(dropped_ambiguous_fine_rows),
        "genres_before_collapse": genres_before_collapse,
        "genres_after_collapse": genres_after_collapse,
        "rows_relabeled_by_collapse": n_relabeled,
        "dropped_exact_duplicate_rows": int(dropped_exact_dupes),
        "dropped_duplicate_track_rows": int(dropped_track_id_dupes),
        "dropped_conflicting_parent_tracks": int(n_ambiguous_tracks),
        "dedupe_strategy": dedupe_strategy,
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
    if report.get("drop_ambiguous_fine_genres"):
        print(
            f"Dropped ambiguous fine genres:    "
            f"{len(report.get('dropped_ambiguous_fine_labels', []))} labels / "
            f"{report.get('dropped_ambiguous_fine_rows', 0):,} rows"
        )
        labels = report.get("dropped_ambiguous_fine_labels", [])
        if labels:
            print(f"  examples: {labels[:12]}{' ...' if len(labels) > 12 else ''}")
    print(
        f"Genres before -> after collapse:  "
        f"{report.get('genres_before_collapse')} -> {report.get('genres_after_collapse')}"
    )
    print(f"Rows relabeled by collapse:       {report.get('rows_relabeled_by_collapse', 0):,}")
    print(f"Dropped exact duplicate rows:     {report['dropped_exact_duplicate_rows']:,}")
    print(f"Dropped duplicate-track rows:     {report['dropped_duplicate_track_rows']:,}")
    if report.get("dropped_conflicting_parent_tracks"):
        print(
            f"Dropped conflicting-parent tracks:"
            f" {report['dropped_conflicting_parent_tracks']:,}"
        )
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
