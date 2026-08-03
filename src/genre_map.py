"""Map fine-grained Spotify genre labels into broader parent genres."""

from __future__ import annotations

# Mood / activity / entertainment / vague tags that are not reliable music genres.
# These rows are dropped during cleaning instead of being forced into a parent.
AMBIGUOUS_FINE_GENRES: frozenset[str] = frozenset(
    {
        # Mood / activity playlists
        "happy",
        "sad",
        "sleep",
        "study",
        "party",
        "chill",
        "romance",
        "new-age",
        # Entertainment / kids / soundtrack-ish grab-bags
        "comedy",
        "disney",
        "show-tunes",
        "children",
        "kids",
        "anime",
        "pop-film",
        # Vague style/region catch-alls
        "guitar",
        "british",
        "indie",
        "alternative",
        "groove",
        # Language tags (not musical genres)
        "french",
        "german",
        "swedish",
        "malay",
        "spanish",
    }
)

# Fine-grained Kaggle labels -> collapsed parent genre (~9 parents).
# Only relatively coherent genre tags are mapped; ambiguous tags are listed above.
GENRE_COLLAPSE_MAP: dict[str, str] = {
    # Pop
    "pop": "pop",
    "power-pop": "pop",
    "synth-pop": "pop",
    "indie-pop": "pop",
    "j-pop": "pop",
    "k-pop": "pop",
    "cantopop": "pop",
    "mandopop": "pop",
    "j-idol": "pop",
    # Rock (includes punk — often confused with rock in audio-feature space)
    "rock": "rock",
    "alt-rock": "rock",
    "psych-rock": "rock",
    "rock-n-roll": "rock",
    "rockabilly": "rock",
    "hard-rock": "rock",
    "grunge": "rock",
    "j-rock": "rock",
    "punk": "rock",
    "punk-rock": "rock",
    "hardcore": "rock",
    "emo": "rock",
    # Metal (kept separate — acoustically distinctive)
    "metal": "metal",
    "heavy-metal": "metal",
    "black-metal": "metal",
    "death-metal": "metal",
    "metalcore": "metal",
    "grindcore": "metal",
    "goth": "metal",
    # Electronic (true electronic/dance + ambient)
    "edm": "electronic",
    "electro": "electronic",
    "electronic": "electronic",
    "house": "electronic",
    "deep-house": "electronic",
    "progressive-house": "electronic",
    "chicago-house": "electronic",
    "techno": "electronic",
    "detroit-techno": "electronic",
    "minimal-techno": "electronic",
    "trance": "electronic",
    "dubstep": "electronic",
    "drum-and-bass": "electronic",
    "idm": "electronic",
    "breakbeat": "electronic",
    "club": "electronic",
    "dance": "electronic",
    "disco": "electronic",
    "hardstyle": "electronic",
    "garage": "electronic",
    "j-dance": "electronic",
    "trip-hop": "electronic",
    "industrial": "electronic",
    "ambient": "electronic",
    # Urban (hip-hop + R&B / soul / funk)
    "hip-hop": "urban",
    "r-n-b": "urban",
    "soul": "urban",
    "funk": "urban",
    # Latin (includes reggae family)
    "latin": "latin",
    "latino": "latin",
    "salsa": "latin",
    "samba": "latin",
    "brazil": "latin",
    "sertanejo": "latin",
    "mpb": "latin",
    "pagode": "latin",
    "forro": "latin",
    "tango": "latin",
    "reggaeton": "latin",
    "reggae": "latin",
    "dancehall": "latin",
    "dub": "latin",
    "ska": "latin",
    # Roots (folk / country / blues / gospel)
    "folk": "roots",
    "singer-songwriter": "roots",
    "songwriter": "roots",
    "acoustic": "roots",
    "country": "roots",
    "honky-tonk": "roots",
    "bluegrass": "roots",
    "blues": "roots",
    "gospel": "roots",
    # Classical / jazz / opera / piano
    "classical": "classical",
    "opera": "classical",
    "piano": "classical",
    "jazz": "classical",
    # World / regional music (language-only and kids tags excluded)
    "afrobeat": "world",
    "indian": "world",
    "iranian": "world",
    "turkish": "world",
    "world-music": "world",
}

# Canonical parent list after the coarse collapse (for reports / docs).
PARENT_GENRES = [
    "classical",
    "electronic",
    "latin",
    "metal",
    "pop",
    "rock",
    "roots",
    "urban",
    "world",
]

# Short examples shown in the README / Streamlit app.
PARENT_GENRE_EXAMPLES: dict[str, str] = {
    "classical": "classical, jazz, opera, piano",
    "electronic": "edm, house, techno, ambient, disco, …",
    "latin": "latin, salsa, reggae, reggaeton, samba, …",
    "metal": "metal, death-metal, metalcore, goth, …",
    "pop": "pop, synth-pop, k-pop, indie-pop, …",
    "rock": "rock, alt-rock, punk, grunge, emo, …",
    "roots": "folk, country, blues, gospel, bluegrass, …",
    "urban": "hip-hop, R&B, soul, funk",
    "world": "afrobeat, indian, turkish, iranian, world-music",
}


def parent_genre_guide_frame():
    """Return a small DataFrame describing each parent prediction label."""
    import pandas as pd

    return pd.DataFrame(
        [
            {"Prediction": name, "Includes (examples)": PARENT_GENRE_EXAMPLES[name]}
            for name in PARENT_GENRES
            if name in PARENT_GENRE_EXAMPLES
        ]
    )


def is_ambiguous_fine_genre(label: str) -> bool:
    """True when a fine-grained label is too noisy to keep for training."""
    return str(label).strip().lower() in AMBIGUOUS_FINE_GENRES


def collapse_genre(label: str) -> str:
    """Return collapsed parent genre, or the original label if unmapped."""
    key = str(label).strip().lower()
    return GENRE_COLLAPSE_MAP.get(key, key)


def collapse_genres_series(genres):
    """Map a pandas Series of genre labels to collapsed parent genres."""
    normalized = genres.astype(str).str.strip().str.lower()
    return normalized.map(GENRE_COLLAPSE_MAP).fillna(normalized)
