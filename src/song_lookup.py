"""Look up songs in the local Spotify tracks dataset and return audio features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_io import AUDIO_FEATURES, RAW_CSV_PATH, TARGET_COLUMN, load_raw_tracks

CATALOG_COLUMNS = [
    c
    for c in (
        "track_id",
        "track_name",
        "artists",
        "album_name",
        TARGET_COLUMN,
        *AUDIO_FEATURES,
    )
]


def catalog_available() -> bool:
    return RAW_CSV_PATH.is_file()


def load_song_catalog() -> pd.DataFrame:
    """
    Load a compact searchable catalog from data/raw/dataset.csv.

    Requires the Kaggle dataset to be present locally.
    """
    if not catalog_available():
        raise FileNotFoundError(
            f"Song lookup needs the dataset at {RAW_CSV_PATH}. "
            "Download it (see README), then re-run lookup."
        )

    df = load_raw_tracks()
    missing = [c for c in ("track_name", "artists", *AUDIO_FEATURES) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns required for lookup: {missing}")

    keep = [c for c in CATALOG_COLUMNS if c in df.columns]
    catalog = df[keep].dropna(subset=["track_name", *AUDIO_FEATURES]).copy()
    catalog["track_name"] = catalog["track_name"].astype(str)
    catalog["artists"] = catalog["artists"].astype(str)
    catalog["_search"] = (
        catalog["track_name"].str.lower() + " " + catalog["artists"].str.lower()
    )
    # Prefer one row per track_id when available
    if "track_id" in catalog.columns:
        catalog = catalog.drop_duplicates(subset=["track_id"], keep="first")
    else:
        catalog = catalog.drop_duplicates(subset=["track_name", "artists"], keep="first")
    return catalog.reset_index(drop=True)


def search_songs(
    query: str,
    catalog: pd.DataFrame | None = None,
    limit: int = 15,
) -> pd.DataFrame:
    """
    Case-insensitive search over track name + artists.

    Returns up to `limit` matches with display + feature columns.
    """
    q = (query or "").strip().lower()
    if len(q) < 2:
        return pd.DataFrame()

    if catalog is None:
        catalog = load_song_catalog()

    tokens = [t for t in q.replace(",", " ").split() if t]
    mask = pd.Series(True, index=catalog.index)
    for token in tokens:
        mask &= catalog["_search"].str.contains(token, regex=False, na=False)

    hits = catalog.loc[mask].copy()
    # Rank: title startswith query first, then shorter titles
    hits["_rank"] = 0
    hits.loc[hits["track_name"].str.lower().str.startswith(q), "_rank"] = 2
    hits.loc[hits["track_name"].str.lower().str.contains(q, regex=False, na=False), "_rank"] += 1
    hits["_title_len"] = hits["track_name"].str.len()
    hits = hits.sort_values(["_rank", "_title_len"], ascending=[False, True])
    hits = hits.head(limit)

    display_cols = [
        c
        for c in ("track_name", "artists", "album_name", TARGET_COLUMN, *AUDIO_FEATURES)
        if c in hits.columns
    ]
    return hits[display_cols].reset_index(drop=True)


def format_match_label(row: pd.Series) -> str:
    title = str(row.get("track_name", "Unknown"))
    artists = str(row.get("artists", "Unknown artist"))
    genre = row.get(TARGET_COLUMN)
    if pd.notna(genre) and str(genre).strip():
        return f"{title} — {artists} [{genre}]"
    return f"{title} — {artists}"


def row_to_features(row: pd.Series) -> dict[str, Any]:
    """Extract model audio-feature dict from a catalog/search row."""
    return {f: row[f] for f in AUDIO_FEATURES}
