"""
Streamlit UI for Spotify genre prediction from audio features.

Run from the project root:
  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_features import extract_features_from_upload, features_to_frame  # noqa: E402
from src.data_io import AUDIO_FEATURES, RAW_CSV_PATH, TARGET_COLUMN  # noqa: E402
from src.evaluate import PREDICTIONS_PATH, predict_genres, predict_topk_table  # noqa: E402
from src.genre_map import parent_genre_guide_frame  # noqa: E402
from src.plots import CONFUSION_MATRIX_NORM_PATH, CONFUSION_MATRIX_PATH  # noqa: E402
from src.song_lookup import (  # noqa: E402
    catalog_available,
    format_match_label,
    load_song_catalog,
    row_to_features,
    search_songs,
)
from src.train import METRICS_PATH, MODEL_PATH, load_metrics, load_trained_model  # noqa: E402

DEFAULT_FEATURES = {
    "danceability": 0.65,
    "energy": 0.70,
    "key": 5,
    "loudness": -7.0,
    "mode": 1,
    "speechiness": 0.06,
    "acousticness": 0.15,
    "instrumentalness": 0.02,
    "liveness": 0.12,
    "valence": 0.55,
    "tempo": 120.0,
    "time_signature": 4,
    "duration_ms": 210000,
}

FEATURE_HELP = {
    "danceability": "How suitable a track is for dancing (0–1).",
    "energy": "Perceptual intensity and activity (0–1).",
    "key": "Estimated pitch class (0=C … 11=B).",
    "loudness": "Overall loudness in dB (typically −60 to 0).",
    "mode": "Modality: 0 = minor, 1 = major.",
    "speechiness": "Presence of spoken words (0–1).",
    "acousticness": "Confidence the track is acoustic (0–1).",
    "instrumentalness": "Likelihood of no vocals (0–1).",
    "liveness": "Presence of a live audience (0–1).",
    "valence": "Musical positiveness (0–1).",
    "tempo": "Estimated tempo in BPM.",
    "time_signature": "Estimated meter (beats per bar).",
    "duration_ms": "Track length in milliseconds.",
}


st.set_page_config(
    page_title="Genre Signal",
    page_icon="♪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap');

      :root {
        --ink: #1c2421;
        --muted: #3f4a45;
        --accent: #0f6b5c;
        --accent-soft: #d8efe9;
        --line: #d9ddd8;
        --panel: #ffffff;
      }

      /* Force readable dark text even if Streamlit/OS is in dark mode */
      .stApp, .stApp * {
        color-scheme: light;
      }

      .stApp {
        background:
          radial-gradient(1200px 500px at 10% -10%, #e7f3ef 0%, transparent 55%),
          linear-gradient(180deg, #f7f5ef 0%, #efece4 100%);
        color: var(--ink) !important;
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
      }

      .stApp p, .stApp span, .stApp label, .stApp li,
      .stApp .stMarkdown, .stApp .stCaption, .stApp .stText,
      .stApp [data-testid="stWidgetLabel"],
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stMetricLabel"],
      .stApp [data-testid="stMetricValue"],
      .stApp [data-testid="stMetricDelta"],
      .stApp [data-baseweb="tab"],
      .stApp [data-baseweb="select"] > div,
      .stApp .stSelectbox, .stApp .stSlider, .stApp .stTextInput,
      .stApp .stFileUploader, .stApp .stAlert {
        color: var(--ink) !important;
      }

      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
      .brand-title {
        font-family: Fraunces, Georgia, serif !important;
        letter-spacing: -0.02em;
        color: var(--ink) !important;
      }

      .hero {
        padding: 0.4rem 0 1.2rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.2rem;
      }

      .brand-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
      }

      .brand-sub {
        color: var(--muted) !important;
        font-size: 1.05rem;
        margin-top: 0.45rem;
        max-width: 48rem;
      }

      .result-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 10px 30px rgba(28, 36, 33, 0.06);
      }

      .genre-pill {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent) !important;
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-weight: 700;
        font-size: 1.15rem;
        text-transform: lowercase;
      }

      .metric-quiet {
        color: var(--muted) !important;
        font-size: 0.95rem;
      }

      .metric-quiet strong {
        color: var(--ink) !important;
      }

      div[data-testid="stSidebar"] {
        background: #fbfaf6;
        border-right: 1px solid var(--line);
      }

      div[data-testid="stSidebar"] * {
        color: var(--ink) !important;
      }

      /* Keep primary buttons readable */
      .stApp .stButton > button[kind="primary"],
      .stApp button[data-testid="baseButton-primary"] {
        color: #ffffff !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_trained_model()


@st.cache_data
def get_metrics():
    if METRICS_PATH.is_file():
        return load_metrics()
    return None


@st.cache_data
def get_sample_tracks(n: int = 40) -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.is_file():
        return None
    df = pd.read_csv(PREDICTIONS_PATH)
    cols = [c for c in AUDIO_FEATURES + ["true_genre", "predicted_genre"] if c in df.columns]
    return df[cols].sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)


@st.cache_data(show_spinner="Loading song catalog…")
def get_catalog() -> pd.DataFrame:
    return load_song_catalog()


def render_prediction(features_df: pd.DataFrame, model, source_note: str | None = None) -> None:
    pred_row = predict_genres(features_df).iloc[0]
    topk = predict_topk_table(features_df, k=5)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f'<span class="genre-pill">{pred_row["predicted_genre"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="metric-quiet" style="margin-top:0.85rem;">'
        f'Confidence: <strong>{float(pred_row["confidence"]):.1%}</strong>'
        f" · {len(model.classes_)} parent genres</p>",
        unsafe_allow_html=True,
    )
    if source_note:
        st.caption(source_note)
    st.markdown("###### Top genre probabilities")
    st.bar_chart(topk.set_index("genre"), horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_feature_controls(defaults: dict) -> dict:
    values = {}
    st.markdown("### Audio features")
    c1, c2 = st.columns(2)
    with c1:
        values["danceability"] = st.slider(
            "danceability", 0.0, 1.0, float(defaults["danceability"]), 0.01,
            help=FEATURE_HELP["danceability"],
        )
        values["energy"] = st.slider(
            "energy", 0.0, 1.0, float(defaults["energy"]), 0.01,
            help=FEATURE_HELP["energy"],
        )
        values["valence"] = st.slider(
            "valence", 0.0, 1.0, float(defaults["valence"]), 0.01,
            help=FEATURE_HELP["valence"],
        )
        values["tempo"] = st.slider(
            "tempo (BPM)", 40.0, 220.0, float(defaults["tempo"]), 0.5,
            help=FEATURE_HELP["tempo"],
        )
        values["loudness"] = st.slider(
            "loudness (dB)", -40.0, 0.0, float(defaults["loudness"]), 0.1,
            help=FEATURE_HELP["loudness"],
        )
        values["acousticness"] = st.slider(
            "acousticness", 0.0, 1.0, float(defaults["acousticness"]), 0.01,
            help=FEATURE_HELP["acousticness"],
        )
        values["duration_ms"] = st.slider(
            "duration (ms)", 30_000, 600_000, int(defaults["duration_ms"]), 1000,
            help=FEATURE_HELP["duration_ms"],
        )
    with c2:
        values["instrumentalness"] = st.slider(
            "instrumentalness", 0.0, 1.0, float(defaults["instrumentalness"]), 0.01,
            help=FEATURE_HELP["instrumentalness"],
        )
        values["speechiness"] = st.slider(
            "speechiness", 0.0, 1.0, float(defaults["speechiness"]), 0.01,
            help=FEATURE_HELP["speechiness"],
        )
        values["liveness"] = st.slider(
            "liveness", 0.0, 1.0, float(defaults["liveness"]), 0.01,
            help=FEATURE_HELP["liveness"],
        )
        values["key"] = st.slider(
            "key", 0, 11, int(defaults["key"]), 1, help=FEATURE_HELP["key"]
        )
        values["mode"] = st.selectbox(
            "mode",
            options=[0, 1],
            index=int(defaults["mode"]),
            format_func=lambda x: "minor (0)" if x == 0 else "major (1)",
            help=FEATURE_HELP["mode"],
        )
        ts_opts = [3, 4, 5, 6, 7]
        ts_default = int(defaults["time_signature"])
        values["time_signature"] = st.selectbox(
            "time_signature",
            options=ts_opts,
            index=ts_opts.index(ts_default) if ts_default in ts_opts else 1,
            help=FEATURE_HELP["time_signature"],
        )
    return values


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <p class="brand-title">Genre Signal</p>
          <p class="brand-sub">
            Predict a track’s parent genre from Spotify audio features.
            Look up a dataset song, upload an MP3/WAV, or set features manually.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.is_file():
        st.error(
            f"Missing trained model at `{MODEL_PATH}`.\n\n"
            "Pull from GitHub or train with `python scripts/train_model.py`."
        )
        st.stop()

    model = get_model()
    metrics = get_metrics()

    with st.sidebar:
        st.markdown("### Parent genres")
        st.caption(
            "The model predicts one of these 9 collapsed labels "
            "(not the fine-grained Kaggle tags)."
        )
        st.dataframe(
            parent_genre_guide_frame(),
            use_container_width=True,
            hide_index=True,
        )

    tab_lookup, tab_upload, tab_manual, tab_model = st.tabs(
        ["Song lookup", "Upload audio", "Manual features", "Model performance"]
    )

    with tab_lookup:
        st.subheader("Look up a song")
        st.markdown(
            "Search by **song title** and/or **artist**. Matches come from the "
            "Kaggle Spotify Tracks dataset used to train this project "
            "(Spotify’s live audio-features API is no longer available for new apps)."
        )

        if not catalog_available():
            st.warning(
                f"Song lookup needs `{RAW_CSV_PATH}`.\n\n"
                "Download the dataset (README Step 2), place `dataset.csv` in "
                "`data/raw/`, then click **Rerun** / refresh this page.\n\n"
                "You can still use the **Manual features** tab without the dataset."
            )
        else:
            try:
                catalog = get_catalog()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load song catalog: {exc}")
                catalog = None

            if catalog is not None:
                st.caption(f"Catalog loaded · {len(catalog):,} searchable tracks")
                with st.form("song_search_form"):
                    query = st.text_input(
                        "Song title or artist",
                        placeholder="e.g. bohemian rhapsody, billie eilish, lo-fi",
                    )
                    submitted = st.form_submit_button("Search", type="primary")

                if submitted:
                    st.session_state["last_query"] = query

                q = st.session_state.get("last_query", "")
                if q and len(str(q).strip()) >= 2:
                    hits = search_songs(str(q), catalog=catalog, limit=20)
                    if hits.empty:
                        st.info("No matches. Try fewer words or a different spelling.")
                    else:
                        labels = [format_match_label(hits.iloc[i]) for i in range(len(hits))]
                        choice = st.selectbox("Matches", options=labels)
                        idx = labels.index(choice)
                        row = hits.iloc[idx]
                        features = row_to_features(row)
                        features_df = pd.DataFrame([features])[AUDIO_FEATURES]

                        meta_cols = [
                            c
                            for c in ("track_name", "artists", "album_name", TARGET_COLUMN)
                            if c in row.index
                        ]
                        st.markdown("###### Selected track")
                        st.write({c: row[c] for c in meta_cols})

                        left, right = st.columns(2)
                        with left:
                            st.markdown("###### Audio features used")
                            st.dataframe(
                                features_df.T.rename(columns={0: "value"}),
                                use_container_width=True,
                            )
                        with right:
                            st.markdown("###### Predicted parent genre")
                            dataset_genre = (
                                str(row[TARGET_COLUMN])
                                if TARGET_COLUMN in row.index and pd.notna(row[TARGET_COLUMN])
                                else None
                            )
                            note = (
                                f"Dataset label for this row: `{dataset_genre}` "
                                "(may be fine-grained; the model predicts a collapsed parent genre)."
                                if dataset_genre
                                else None
                            )
                            render_prediction(features_df, model, source_note=note)
                elif submitted:
                    st.info("Enter at least 2 characters to search.")

    with tab_upload:
        st.subheader("Upload an audio file")
        st.markdown(
            "Upload an **MP3, WAV, FLAC, or OGG** file from outside the dataset. "
            "The app estimates Spotify-like audio features with **librosa**, then "
            "runs the trained model.\n\n"
            "**Note:** These estimates are approximate (not Spotify’s proprietary "
            "features), so predictions may be less accurate than dataset lookup."
        )
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "flac", "ogg", "m4a"],
        )
        analyze = st.button(
            "Extract features & predict",
            type="primary",
            disabled=uploaded is None,
        )
        if uploaded is not None and analyze:
            with st.spinner("Analyzing audio (first ~90 seconds)…"):
                try:
                    feats = extract_features_from_upload(
                        uploaded.getvalue(),
                        filename=uploaded.name,
                    )
                    features_df = features_to_frame(feats)
                    st.session_state["upload_features"] = features_df
                    st.session_state["upload_name"] = uploaded.name
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        f"Could not read/analyze this file: {exc}\n\n"
                        "Tip: WAV usually works without extra setup. "
                        "MP3 often requires **ffmpeg** installed on your system."
                    )
                    st.session_state.pop("upload_features", None)

        if "upload_features" in st.session_state:
            features_df = st.session_state["upload_features"]
            st.caption(f"File: `{st.session_state.get('upload_name', 'upload')}`")
            left, right = st.columns(2)
            with left:
                st.markdown("###### Estimated features")
                st.dataframe(
                    features_df.T.rename(columns={0: "value"}),
                    use_container_width=True,
                )
            with right:
                st.markdown("###### Predicted parent genre")
                render_prediction(
                    features_df,
                    model,
                    source_note=(
                        "Features were estimated from the audio file (approximate). "
                        "For exact Spotify features, use Song lookup on dataset tracks."
                    ),
                )

    with tab_manual:
        st.subheader("Predict from manual audio features")
        samples = get_sample_tracks()
        defaults = st.session_state.get("feature_defaults", DEFAULT_FEATURES.copy())

        if samples is not None and len(samples):
            with st.expander("Load a held-out test example"):
                options = ["(manual)"] + [f"row {i}" for i in range(len(samples))]
                pick = st.selectbox("Example", options=options, key="manual_sample_pick")
                if pick != "(manual)":
                    idx = int(pick.split()[-1])
                    row = samples.iloc[idx]
                    defaults = {f: row[f] for f in AUDIO_FEATURES}
                    st.caption(
                        f"True genre: {row.get('true_genre', '?')} · "
                        f"Saved model prediction: {row.get('predicted_genre', '?')}"
                    )

        feature_values = render_feature_controls(defaults)
        features_df = pd.DataFrame([feature_values])[AUDIO_FEATURES]
        left, right = st.columns(2)
        with left:
            st.dataframe(features_df.T.rename(columns={0: "value"}), use_container_width=True)
            run = st.button("Predict genre", type="primary", use_container_width=True)
        with right:
            if run:
                render_prediction(features_df, model)
            else:
                st.info("Adjust features, then click **Predict genre**.")

    with tab_model:
        st.subheader("Held-out test performance")
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
            c3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}")
            c4.metric("Genres", f"{metrics.get('n_classes', len(model.classes_))}")
            st.caption(
                "Metrics use a stratified 20% test split after cleaning and "
                "collapsing fine-grained genres into parent labels."
            )
        else:
            st.warning("No `models/metrics.json` found.")

        st.subheader("Confusion matrices")
        img_cols = st.columns(2)
        with img_cols[0]:
            if CONFUSION_MATRIX_PATH.is_file():
                st.image(str(CONFUSION_MATRIX_PATH), caption="Raw counts")
            else:
                st.caption("Missing outputs/confusion_matrix.png")
        with img_cols[1]:
            if CONFUSION_MATRIX_NORM_PATH.is_file():
                st.image(str(CONFUSION_MATRIX_NORM_PATH), caption="Row-normalized")
            else:
                st.caption("Missing outputs/confusion_matrix_normalized.png")

        with st.expander("What each predicted genre means", expanded=True):
            st.caption(
                "Dataset song lookup may still show a fine-grained Kaggle tag; "
                "the model always predicts one of these parent genres."
            )
            st.dataframe(
                parent_genre_guide_frame(),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
