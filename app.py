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

from src.data_io import AUDIO_FEATURES  # noqa: E402
from src.evaluate import PREDICTIONS_PATH, predict_genres, predict_topk_table  # noqa: E402
from src.plots import CONFUSION_MATRIX_NORM_PATH, CONFUSION_MATRIX_PATH  # noqa: E402
from src.train import METRICS_PATH, MODEL_PATH, load_metrics, load_trained_model  # noqa: E402

# Sensible defaults near mid-range pop/electronic values
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
        --muted: #5b6762;
        --paper: #f3f1ea;
        --panel: #ffffff;
        --accent: #0f6b5c;
        --accent-soft: #d8efe9;
        --line: #d9ddd8;
      }

      .stApp {
        background:
          radial-gradient(1200px 500px at 10% -10%, #e7f3ef 0%, transparent 55%),
          linear-gradient(180deg, #f7f5ef 0%, #efece4 100%);
        color: var(--ink);
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
      }

      h1, h2, h3, .brand-title {
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
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 0.45rem;
        max-width: 46rem;
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
        color: var(--accent);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-weight: 700;
        font-size: 1.15rem;
        text-transform: lowercase;
      }

      .metric-quiet {
        color: var(--muted);
        font-size: 0.95rem;
      }

      div[data-testid="stSidebar"] {
        background: #fbfaf6;
        border-right: 1px solid var(--line);
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


def render_feature_controls(defaults: dict) -> dict:
    values = {}
    st.sidebar.markdown("### Audio features")
    st.sidebar.caption(
        "The model predicts from Spotify-style audio features — not from a song title alone."
    )

    values["danceability"] = st.sidebar.slider(
        "danceability", 0.0, 1.0, float(defaults["danceability"]), 0.01,
        help=FEATURE_HELP["danceability"],
    )
    values["energy"] = st.sidebar.slider(
        "energy", 0.0, 1.0, float(defaults["energy"]), 0.01,
        help=FEATURE_HELP["energy"],
    )
    values["valence"] = st.sidebar.slider(
        "valence", 0.0, 1.0, float(defaults["valence"]), 0.01,
        help=FEATURE_HELP["valence"],
    )
    values["tempo"] = st.sidebar.slider(
        "tempo (BPM)", 40.0, 220.0, float(defaults["tempo"]), 0.5,
        help=FEATURE_HELP["tempo"],
    )
    values["loudness"] = st.sidebar.slider(
        "loudness (dB)", -40.0, 0.0, float(defaults["loudness"]), 0.1,
        help=FEATURE_HELP["loudness"],
    )
    values["acousticness"] = st.sidebar.slider(
        "acousticness", 0.0, 1.0, float(defaults["acousticness"]), 0.01,
        help=FEATURE_HELP["acousticness"],
    )
    values["instrumentalness"] = st.sidebar.slider(
        "instrumentalness", 0.0, 1.0, float(defaults["instrumentalness"]), 0.01,
        help=FEATURE_HELP["instrumentalness"],
    )
    values["speechiness"] = st.sidebar.slider(
        "speechiness", 0.0, 1.0, float(defaults["speechiness"]), 0.01,
        help=FEATURE_HELP["speechiness"],
    )
    values["liveness"] = st.sidebar.slider(
        "liveness", 0.0, 1.0, float(defaults["liveness"]), 0.01,
        help=FEATURE_HELP["liveness"],
    )
    values["duration_ms"] = st.sidebar.slider(
        "duration (ms)", 30_000, 600_000, int(defaults["duration_ms"]), 1000,
        help=FEATURE_HELP["duration_ms"],
    )
    values["key"] = st.sidebar.slider(
        "key", 0, 11, int(defaults["key"]), 1,
        help=FEATURE_HELP["key"],
    )
    values["mode"] = st.sidebar.selectbox(
        "mode",
        options=[0, 1],
        index=int(defaults["mode"]),
        format_func=lambda x: "minor (0)" if x == 0 else "major (1)",
        help=FEATURE_HELP["mode"],
    )
    values["time_signature"] = st.sidebar.selectbox(
        "time_signature",
        options=[3, 4, 5, 6, 7],
        index=[3, 4, 5, 6, 7].index(int(defaults["time_signature"]))
        if int(defaults["time_signature"]) in [3, 4, 5, 6, 7]
        else 1,
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
            Adjust the feature controls, or load a sample from the held-out test set.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.is_file():
        st.error(
            f"Missing trained model at `{MODEL_PATH}`.\n\n"
            "Train first with `python scripts/train_model.py`, or pull the model from GitHub."
        )
        st.stop()

    model = get_model()
    metrics = get_metrics()
    samples = get_sample_tracks()

    if "feature_defaults" not in st.session_state:
        st.session_state.feature_defaults = DEFAULT_FEATURES.copy()

    with st.sidebar:
        st.markdown("### Load a sample track")
        if samples is not None and len(samples):
            options = ["(manual features)"] + [f"row {i}" for i in range(len(samples))]
            pick = st.selectbox("Test-set example", options=options, key="sample_pick")
            if pick != st.session_state.get("_applied_sample_pick"):
                st.session_state._applied_sample_pick = pick
                if pick == "(manual features)":
                    st.session_state.feature_defaults = DEFAULT_FEATURES.copy()
                else:
                    idx = int(pick.split()[-1])
                    row = samples.iloc[idx]
                    st.session_state.feature_defaults = {f: row[f] for f in AUDIO_FEATURES}
                    st.info(
                        f"Loaded sample · true genre: **{row.get('true_genre', '?')}**"
                        f" · model had predicted: **{row.get('predicted_genre', '?')}**"
                    )
                    st.rerun()
            if st.button("Reset to defaults", use_container_width=True):
                st.session_state.feature_defaults = DEFAULT_FEATURES.copy()
                st.session_state._applied_sample_pick = "(manual features)"
                st.session_state.sample_pick = "(manual features)"
                st.rerun()
        else:
            st.caption("No `outputs/test_predictions.csv` found for samples.")

    feature_values = render_feature_controls(st.session_state.feature_defaults)
    features_df = pd.DataFrame([feature_values])[AUDIO_FEATURES]

    tab_predict, tab_model = st.tabs(["Predict", "Model performance"])

    with tab_predict:
        left, right = st.columns([1.05, 1.0], gap="large")

        with left:
            st.subheader("Current features")
            st.dataframe(features_df.T.rename(columns={0: "value"}), use_container_width=True)
            run = st.button("Predict genre", type="primary", use_container_width=True)

        with right:
            st.subheader("Prediction")
            if run or "last_prediction" in st.session_state:
                if run:
                    pred_row = predict_genres(features_df).iloc[0]
                    topk = predict_topk_table(features_df, k=5)
                    st.session_state.last_prediction = {
                        "genre": str(pred_row["predicted_genre"]),
                        "confidence": float(pred_row["confidence"]),
                        "topk": topk,
                    }

                result = st.session_state.last_prediction
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="genre-pill">{result["genre"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="metric-quiet" style="margin-top:0.85rem;">'
                    f'Confidence: <strong>{result["confidence"]:.1%}</strong>'
                    f" · {len(model.classes_)} parent genres</p>",
                    unsafe_allow_html=True,
                )
                st.markdown("###### Top genre probabilities")
                chart_df = result["topk"].set_index("genre")
                st.bar_chart(chart_df, horizontal=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Set features in the sidebar, then click **Predict genre**.")

    with tab_model:
        st.subheader("Held-out test performance")
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
            c3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}")
            c4.metric("Genres", f"{metrics.get('n_classes', len(model.classes_))}")
            st.caption(
                "Metrics are from the stratified 20% test split after cleaning "
                "and collapsing fine-grained genres into parent labels."
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
                st.image(
                    str(CONFUSION_MATRIX_NORM_PATH),
                    caption="Row-normalized",
                )
            else:
                st.caption("Missing outputs/confusion_matrix_normalized.png")

        with st.expander("Model classes"):
            st.write(", ".join(str(c) for c in model.classes_))


if __name__ == "__main__":
    main()
