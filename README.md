# Spotify Genre Prediction (Genre Signal)

## Research question

> How accurately can a supervised machine learning model predict a song’s genre from its Spotify audio features alone?

This repository trains a classifier on audio features from the public
[Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset),
evaluates it on a held-out test set, and provides a **Streamlit app** so users can:

1. **Look up a song** (by title/artist) in the project dataset and get a predicted parent genre  
2. **Upload an MP3/WAV** to estimate features for songs outside the dataset  
3. **Manually set audio features** and predict  
4. View **test metrics** and **confusion matrices**

**Repo:** https://github.com/ada406/SU26-7030-Capstone-

**Current held-out test accuracy:** about **59.1%** on **9 parent genres**  
(random baseline ≈ 11%; see `models/metrics.json` for the exact number)

---

## Do graders need the dataset?

| Goal | Need `data/raw/dataset.csv`? |
|------|------------------------------|
| Rebuild conda env + run Streamlit | **No** |
| **Manual features**, **Upload audio**, **Model performance** | **No** (model + metrics + figures are in the repo) |
| **Song lookup** | **Yes** |
| Retrain / re-evaluate from scratch | **Yes** |

On a clean checkout you can demonstrate the main result without Kaggle: open the app, inspect metrics, and view the committed confusion matrices.

---

## Quick start for graders (recommended)

These steps assume a laptop/desktop with conda (Miniconda, Miniforge, or Anaconda).  
OSC OnDemand also works; see notes near the end.

### 1) Clone

```bash
git clone https://github.com/ada406/SU26-7030-Capstone-.git
cd SU26-7030-Capstone-
```

### 2) Create and activate the environment

```bash
conda env create -f environment.yml
conda activate spotify-genre-prediction
```

If the env already exists:

```bash
conda env update -f environment.yml --prune
conda activate spotify-genre-prediction
```

The environment includes `streamlit`, scikit-learn, and audio deps (`librosa`, `ffmpeg`, etc.) for the upload tab.

### 3) Run the Streamlit app (no dataset required)

```bash
conda activate spotify-genre-prediction
streamlit run app.py
```

Open the Local URL shown in the terminal (usually http://localhost:8501).  
Keep that terminal open while using the app.

Suggested grader checks without data:

1. **Model performance** — accuracy ~59%, F1 scores, confusion matrices  
2. **Manual features** — adjust sliders → **Predict genre**  
3. Sidebar — 9 parent-genre guide  

### 4) Optional: download the dataset (song lookup / retrain only)

1. Open https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset  
2. Download the ZIP (login + accept terms if prompted)  
3. Place the ZIP or `dataset.csv` in `data/raw/`

```bash
mkdir -p data/raw
# after copying the file into data/raw/:
python scripts/download_data.py   # unzips / normalizes to data/raw/dataset.csv
python scripts/load_data.py       # optional sanity check
```

---

## Results / findings

**Answer to the research question:** Using only Spotify-style audio features, a Random Forest can predict a song’s **parent genre** substantially better than chance, but not with high precision.

| Metric | Value |
|--------|--------|
| Held-out test accuracy | **~59.1%** |
| Macro F1 | **~0.55** |
| Classes | **9** parent genres |
| Random baseline | ≈ **11%** (1/9) |
| Train / test | stratified 80 / 20 (`random_state=42`) |

**What works well:** `electronic`, `classical`, and `metal` are the strongest classes (more distinctive acoustic profiles).  

**What is hard:** `world`, `rock`, `pop`, and `urban` overlap a lot in feature space, so they are confused more often.  

**Why this matters:** Genre is partly cultural/marketing, not a pure acoustic category. With 13 numeric audio features alone, mid/high-50s to low-60s accuracy on ~9 parents is a realistic outcome. Collapsing 114 fine tags into 9 cleaner parents, dropping mood/vague labels, and removing tracks with conflicting parents improved accuracy versus an earlier ~17-genre setup (~47%).

### Main result visualization

The primary figure is the **row-normalized confusion matrix**:

![Normalized confusion matrix](outputs/confusion_matrix_normalized.png)

- **File:** `outputs/confusion_matrix_normalized.png`  
- **How to read it:** rows = true parent genre, columns = predicted genre; each row sums to ~1.  
- **What it shows:** a visible diagonal (the model has real signal) plus off-diagonal mass among overlapping genres (especially pop / rock / world / urban).  
- A raw-count version is at `outputs/confusion_matrix.png`. Both also appear in the Streamlit **Model performance** tab.

---

## Using the Streamlit app

| Tab | What to do | Needs dataset? |
|-----|------------|----------------|
| **Song lookup** | Type a title/artist → **Search** → pick a match → predicted parent genre | Yes |
| **Upload audio** | Upload MP3/WAV/FLAC/OGG → extract features → predict | No |
| **Manual features** | Move sliders → **Predict genre** | No |
| **Model performance** | View accuracy / F1, confusion matrices, genre guide | No |

**Sidebar:** lists the 9 parent genres the model can predict and what each includes.

### Important: dataset label vs model prediction

- The **Kaggle CSV is not rewritten**. Song lookup still shows the original fine-grained tag (e.g. `synth-pop`).  
- The **model only predicts one of 9 parent genres** (e.g. `pop`).  
- So a track labeled `synth-pop` in the dataset may correctly be predicted as `pop`. That is expected.

**Song lookup limitation:** Spotify removed public access to the live Audio Features API for new apps (2024).  
Lookup therefore searches **inside the Kaggle dataset** used for training — not the live Spotify catalog.

**Upload audio limitation:** Features are estimated with `librosa` (not Spotify’s proprietary values), so predictions for arbitrary files are approximate. MP3 decoding needs **ffmpeg** (included in the conda env). WAV is the most reliable format.

---

## What the model predicts (9 parent genres)

Fine-grained Kaggle tags (114) are collapsed into these parents during training (`src/genre_map.py`):

| Prediction | Includes (examples) |
|---|---|
| **classical** | classical, jazz, opera, piano |
| **electronic** | edm, house, techno, ambient, disco, … |
| **latin** | latin, salsa, reggae, reggaeton, samba, … |
| **metal** | metal, death-metal, metalcore, goth, … |
| **pop** | pop, synth-pop, k-pop, indie-pop, … |
| **rock** | rock, alt-rock, punk, grunge, emo, … |
| **roots** | folk, country, blues, gospel, bluegrass, … |
| **urban** | hip-hop, R&B, soul, funk |
| **world** | afrobeat, indian, turkish, iranian, world-music |

Mood / vague / entertainment tags (e.g. `happy`, `sleep`, `comedy`, `kids`) are **dropped** before training, and tracks that map to **conflicting parents** are removed so labels stay cleaner.

---

## What is already included in the repo

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI (“Genre Signal”) |
| `environment.yml` | Reproducible conda environment |
| `models/genre_classifier.joblib` | Trained Random Forest pipeline (~60 MB) |
| `models/metrics.json` | Test-set metrics + training settings |
| `models/feature_columns.json` | Feature list used by the model |
| `outputs/test_predictions.csv` | Held-out predictions |
| `outputs/confusion_matrix.png` | Confusion matrix (counts) |
| `outputs/confusion_matrix_normalized.png` | **Main result figure** (row-normalized) |
| `scripts/` | **Canonical** download / train / evaluate / plot entrypoints |
| `notebooks/` | Same workflows interactively (wrappers around `src/` + scripts) |
| `src/` | Shared library code (`genre_map`, `clean`, `train`, …) |

**Not committed (by design):** `data/raw/dataset.csv` (download from Kaggle).

**Canonical pipeline:** prefer `scripts/*.py` for reproducibility. Notebooks call the same library code and should match the scripts; if anything diverges, trust the scripts + committed `models/` / `outputs/`.

---

## Repository structure

```text
SU26-7030-Capstone-/
├── app.py
├── environment.yml
├── README.md
├── data/
│   ├── raw/                 # put dataset.csv here (not in git)
│   └── processed/           # created when training/cleaning
├── models/                  # trained model + metrics
├── outputs/                 # predictions + figures
├── notebooks/
├── scripts/
└── src/
```

---

## Modeling summary

- **Data:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~114k tracks, 114 fine-grained `track_genre` labels)  
- **Features:** 13 Spotify audio features (`danceability`, `energy`, `tempo`, `acousticness`, …)  
- **Labels:** collapsed to the **9 parent genres** in the table above  
- **Cleaning:** drop ambiguous fine tags; drop multi-parent conflicts; drop NA/duplicates; drop rare parents  
- **Split:** stratified **80% train / 20% test** (`random_state=42`)  
- **Model:** `StandardScaler` + `RandomForestClassifier`  
  (size-capped tune: `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`, `max_features=0.5`)  
- **Reported test accuracy:** ~**59.1%** on 9 classes (see `models/metrics.json`; random baseline ≈ 11%)

---

## Full pipeline (optional — model already trained)

Use this only if you want to rebuild artifacts from scratch. Requires `data/raw/dataset.csv`.

```bash
python scripts/download_data.py
python scripts/load_data.py
python scripts/train_model.py --model rf --no-tune
# Or compare RF vs HistGradientBoosting with tuning:
#   python scripts/train_model.py --model compare --tune
python scripts/evaluate_model.py
python scripts/plot_confusion_matrix.py
streamlit run app.py
```

The committed `models/genre_classifier.joblib` already reflects the cleaned 9-genre Random Forest above; graders do **not** need to retrain to use the app or inspect the main result.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_load_data.ipynb` | Download/load validation |
| `notebooks/02_train_model.ipynb` | Clean + train (9 parents; matches script defaults) |
| `notebooks/03_evaluate_predict.ipynb` | Evaluate + predict |
| `notebooks/04_confusion_matrix.ipynb` | Confusion matrices (main result figure) |

Use the `spotify-genre-prediction` kernel when prompted.  
Again: **`scripts/` is canonical** for a clean command-line rerun.

---

## OSC OnDemand notes

```bash
module load miniconda3/24.1.2-py310   # module name may vary
cd ~/SU26-7030-Capstone-
git pull
conda activate spotify-genre-prediction
# if streamlit missing:
#   conda install -y streamlit
streamlit run app.py --server.port 8502 --server.address 127.0.0.1
```

Opening the app in a browser on OSC may require SSH port forwarding.  
For the simplest demo, run `streamlit run app.py` on a local machine after cloning.

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| `conda: command not found` | Install Miniconda/Miniforge, reopen terminal, or `source ~/miniforge3/bin/activate` |
| `Port 8501 is not available` | `streamlit run app.py --server.port 8502` |
| Song lookup warns missing CSV | Put `dataset.csv` in `data/raw/` and refresh the app (optional for other tabs) |
| Upload audio fails on MP3 | Confirm `which ffmpeg` after `conda activate`, or upload WAV |
| `No module named 'librosa'` | `conda env update -f environment.yml --prune` (or recreate the env) |
| `No module named 'streamlit'` | `conda install streamlit` or `pip install streamlit` inside the env |
| White / hard-to-read text | Refresh the app; current CSS forces dark text on the light theme |
| `ERR_CONNECTION_REFUSED` in browser | Streamlit isn’t running — start it and keep that terminal open |
| Want to recreate env cleanly | `conda env remove -n spotify-genre-prediction` then `conda env create -f environment.yml` |

---

## License / data credit

- Dataset: [Maharshi Pandya — Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)  
- Follow Kaggle’s and the dataset’s terms when sharing results or derivatives.
