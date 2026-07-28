"""Map fine-grained Spotify genre labels into broader parent genres."""

from __future__ import annotations

# Fine-grained Kaggle labels -> collapsed parent genre.
# Unmapped labels keep their original name (then rare-genre filtering may drop them).
GENRE_COLLAPSE_MAP: dict[str, str] = {
    # Pop
    "pop": "pop",
    "pop-film": "pop",
    "power-pop": "pop",
    "synth-pop": "pop",
    "indie-pop": "pop",
    "j-pop": "pop",
    "k-pop": "pop",
    "cantopop": "pop",
    "mandopop": "pop",
    "j-idol": "pop",
    # Rock
    "rock": "rock",
    "alt-rock": "rock",
    "alternative": "rock",
    "psych-rock": "rock",
    "rock-n-roll": "rock",
    "rockabilly": "rock",
    "hard-rock": "rock",
    "grunge": "rock",
    "j-rock": "rock",
    "guitar": "rock",
    "british": "rock",
    "indie": "rock",
    # Metal
    "metal": "metal",
    "heavy-metal": "metal",
    "black-metal": "metal",
    "death-metal": "metal",
    "metalcore": "metal",
    "grindcore": "metal",
    "industrial": "metal",
    "goth": "metal",
    # Punk / hardcore
    "punk": "punk",
    "punk-rock": "punk",
    "hardcore": "punk",
    "emo": "punk",
    # Electronic / dance
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
    # Hip-hop / rap
    "hip-hop": "hip-hop",
    # R&B / soul / funk
    "r-n-b": "r&b",
    "soul": "r&b",
    "funk": "r&b",
    "groove": "r&b",
    # Jazz
    "jazz": "jazz",
    # Classical / opera / piano
    "classical": "classical",
    "opera": "classical",
    "piano": "classical",
    "romance": "classical",
    # Country
    "country": "country",
    "honky-tonk": "country",
    "bluegrass": "country",
    # Folk / singer-songwriter / acoustic
    "folk": "folk",
    "singer-songwriter": "folk",
    "songwriter": "folk",
    "acoustic": "folk",
    # Latin / Brazilian
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
    "spanish": "latin",
    # Reggae family
    "reggae": "reggae",
    "dancehall": "reggae",
    "dub": "reggae",
    "ska": "reggae",
    # Blues
    "blues": "blues",
    # Ambient / chill / sleep / study
    "ambient": "ambient",
    "chill": "ambient",
    "sleep": "ambient",
    "study": "ambient",
    "new-age": "ambient",
    "happy": "ambient",
    "sad": "ambient",
    # World / regional
    "afrobeat": "world",
    "indian": "world",
    "iranian": "world",
    "turkish": "world",
    "malay": "world",
    "french": "world",
    "german": "world",
    "swedish": "world",
    "world-music": "world",
    # Kids / family entertainment
    "children": "kids",
    "kids": "kids",
    "disney": "kids",
    "show-tunes": "kids",
    "comedy": "kids",
    "anime": "kids",
    "party": "pop",
}


def collapse_genre(label: str) -> str:
    """Return collapsed parent genre, or the original label if unmapped."""
    key = str(label).strip().lower()
    return GENRE_COLLAPSE_MAP.get(key, key)


def collapse_genres_series(genres):
    """Map a pandas Series of genre labels to collapsed parent genres."""
    normalized = genres.astype(str).str.strip().str.lower()
    return normalized.map(GENRE_COLLAPSE_MAP).fillna(normalized)
