"""
Estimate Spotify-like audio features from an uploaded audio file.

These are heuristics based on librosa signal processing. They are NOT identical
to Spotify's proprietary audio features, so model predictions for uploaded files
are approximate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.data_io import AUDIO_FEATURES

# Cap analysis length for responsiveness in the Streamlit app
MAX_ANALYZE_SECONDS = 90.0
TARGET_SR = 22050


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _estimate_key_and_mode(chroma: np.ndarray) -> tuple[int, int]:
    """
    Estimate pitch class key (0-11) and mode (0=minor, 1=major) from chroma.

    Uses correlation with simple major/minor profiles (Krumhansl-style lite).
    """
    # Average chroma over time
    chroma_mean = np.mean(chroma, axis=1)
    if chroma_mean.sum() <= 0:
        return 0, 1

    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    best_key = 0
    best_mode = 1
    best_score = -np.inf
    for shift in range(12):
        maj = float(np.corrcoef(np.roll(major_profile, shift), chroma_mean)[0, 1])
        mi = float(np.corrcoef(np.roll(minor_profile, shift), chroma_mean)[0, 1])
        if np.isnan(maj):
            maj = -1.0
        if np.isnan(mi):
            mi = -1.0
        if maj > best_score:
            best_score = maj
            best_key = shift
            best_mode = 1
        if mi > best_score:
            best_score = mi
            best_key = shift
            best_mode = 0
    return int(best_key), int(best_mode)


def extract_features_from_array(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Compute approximate Spotify-like features from a mono waveform."""
    import librosa

    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = np.ascontiguousarray(y, dtype=np.float32)

    duration_s = float(len(y) / sr) if sr else 0.0
    duration_ms = int(round(duration_s * 1000))

    # Dynamics / energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms) + 1e-9)
    energy = _clip01(rms_mean * 4.5)
    loudness = float(np.clip(20.0 * np.log10(rms_mean), -60.0, 0.0))

    # Rhythm
    tempo_arr = librosa.beat.tempo(y=y, sr=sr)
    tempo = float(np.clip(tempo_arr[0] if len(tempo_arr) else 120.0, 40.0, 220.0))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = float(np.mean(onset_env))
    tempo_factor = 1.0 - abs(tempo - 120.0) / 120.0
    danceability = _clip01(0.45 * _clip01(tempo_factor) + 0.55 * _clip01(onset_rate / 3.0))

    # Timbre / spectrum
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = float(np.mean(contrast))

    # Acousticness: lower centroid / brightness → more acoustic-like
    acousticness = _clip01(1.2 - (cent / 4000.0) - 0.4 * flatness)

    # Speechiness: higher ZCR + flatness often correlates with speech-like signals
    speechiness = _clip01(0.15 + 2.2 * zcr + 0.35 * flatness - 0.2 * energy)

    # Instrumentalness proxy via harmonic/percussive balance + spectral stability
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_ratio = float(np.sum(np.abs(y_harm)) / (np.sum(np.abs(y)) + 1e-9))
    perc_ratio = float(np.sum(np.abs(y_perc)) / (np.sum(np.abs(y)) + 1e-9))
    instrumentalness = _clip01(0.15 + 0.7 * harm_ratio - 0.35 * speechiness)

    # Liveness proxy: more percussive / contrasty / noisy → slightly higher
    liveness = _clip01(0.08 + 0.55 * perc_ratio + 0.15 * _clip01(contrast_mean / 30.0))

    # Valence proxy: brighter + more danceable + major mode later
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode = _estimate_key_and_mode(chroma)
    brightness = _clip01(cent / 3500.0)
    valence = _clip01(
        0.25 * danceability + 0.35 * brightness + 0.20 * (1.0 if mode == 1 else 0.35) + 0.20 * energy
    )

    # Time signature: default 4; crude beat-interval heuristic otherwise
    time_signature = 4
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        if len(beats) >= 8:
            intervals = np.diff(beats)
            med = float(np.median(intervals))
            # Prefer common meters; keep simple
            if 0.35 <= med <= 0.55:
                time_signature = 4
            elif med < 0.35:
                time_signature = 3
            else:
                time_signature = 4
    except Exception:  # noqa: BLE001
        time_signature = 4

    features = {
        "danceability": danceability,
        "energy": energy,
        "key": int(key),
        "loudness": loudness,
        "mode": int(mode),
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "time_signature": int(time_signature),
        "duration_ms": int(duration_ms),
    }
    # Ensure column order / completeness
    return {name: features[name] for name in AUDIO_FEATURES}


def extract_features_from_file(
    path: str | Path,
    max_seconds: float = MAX_ANALYZE_SECONDS,
) -> dict[str, Any]:
    """Load an audio file from disk and extract approximate features."""
    import librosa

    y, sr = librosa.load(
        str(path),
        sr=TARGET_SR,
        mono=True,
        duration=max_seconds if max_seconds > 0 else None,
    )
    return extract_features_from_array(y, sr)


def extract_features_from_upload(
    file_bytes: bytes,
    filename: str = "upload.wav",
    max_seconds: float = MAX_ANALYZE_SECONDS,
) -> dict[str, Any]:
    """
    Extract features from an uploaded file's raw bytes.

    Writes to a temporary file because some backends (MP3 via audioread/ffmpeg)
    need a real path.
    """
    suffix = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        return extract_features_from_file(tmp_path, max_seconds=max_seconds)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def features_to_frame(features: dict[str, Any]):
    import pandas as pd

    return pd.DataFrame([{name: features[name] for name in AUDIO_FEATURES}])
