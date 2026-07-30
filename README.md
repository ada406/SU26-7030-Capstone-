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

---

## Quick start for graders (recommended)

These steps assume a laptop/desktop with conda (Miniconda, Miniforge, or Anaconda).
OSC OnDemand also works; see notes at the end.

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

### 3) Download the dataset (needed for song lookup + optional retrain)

The trained model is already in `models/`, so the app’s **Manual features** and
**Model performance** tabs work without data. **Song lookup** needs the CSV.

1. Open https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset  
2. Download the ZIP (login + accept terms if prompted)  
3. Place the ZIP or `dataset.csv` in `data/raw/`

```bash
mkdir -p data/raw
# after copying the file into data/raw/:
python scripts/download_data.py   # unzips / normalizes to data/raw/dataset.csv
python scripts/load_data.py       # optional sanity check
```

### 4) Run the Streamlit app

```bash
conda activate spotify-genre-prediction
streamlit run app.py
```

Open the Local URL shown in the terminal (usually http://localhost:8501).

#### Using the app

| Tab | What to do |
|-----|------------|
| **Song lookup** | Type a title/artist → **Search** → pick a match → predicted genre |
| **Upload audio** | Upload MP3/WAV → extract approximate features → predict |
| **Manual features** | Move sliders → **Predict genre** |
| **Model performance** | View accuracy / F1 and confusion matrix images |

**Song lookup limitation:** Spotify removed public access to the live Audio Features API for new apps (2024).  
Lookup therefore searches **inside the Kaggle dataset** used for training.

**Upload audio limitation:** Features are estimated with `librosa` (not Spotify’s proprietary values), so predictions for arbitrary MP3s are approximate. MP3 decoding often requires **ffmpeg** installed on your machine; WAV is the most reliable format.

---

## What is already included in the repo

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `environment.yml` | Reproducible conda environment |
| `models/genre_classifier.joblib` | Trained model (~71 MB) |
| `models/metrics.json` | Test-set metrics (~47.5% accuracy, 17 parent genres) |
| `models/feature_columns.json` | Feature list used by the model |
| `outputs/test_predictions.csv` | Held-out predictions |
| `outputs/confusion_matrix.png` | Confusion matrix (counts) |
| `outputs/confusion_matrix_normalized.png` | Confusion matrix (row-normalized) |
| `scripts/` | Download, train, evaluate, plot |
| `notebooks/` | Same workflows interactively |
| `src/` | Shared library code |

**Not committed (by design):** `data/raw/dataset.csv` (download from Kaggle).

---

## Repository structure

```text
SU26-7030-Capstone-/
├── app.py
├── environment.yml
├── README.md
├── data/
│   ├── raw/                 # put dataset.csv here
│   └── processed/           # created when training/cleaning
├── models/                  # trained model + metrics
├── outputs/                 # predictions + figures
├── notebooks/
├── scripts/
└── src/
```

---

## Full pipeline (optional — model already trained)

From the project root, with the env activated and `data/raw/dataset.csv` present:

```bash
python scripts/download_data.py
python scripts/load_data.py
python scripts/train_model.py          # clean, collapse genres, 80/20 split, save model
python scripts/evaluate_model.py       # score test set → outputs/test_predictions.csv
python scripts/plot_confusion_matrix.py
streamlit run app.py
```

### Modeling summary

- **Features:** 13 Spotify audio features (`danceability`, `energy`, `tempo`, …)  
- **Labels:** fine-grained genres collapsed into ~17 parent genres (`src/genre_map.py`)  
- **Cleaning:** drop NA/duplicates; one row per track (prefer common genres); drop rare parents  
- **Split:** stratified **80% train / 20% test** (`random_state=42`)  
- **Model:** `StandardScaler` + depth-limited `RandomForestClassifier`  
- **Reported test accuracy:** see `models/metrics.json` (about **47.5%** on 17 classes; random baseline ≈ 6%)

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_load_data.ipynb` | Download/load validation |
| `notebooks/02_train_model.ipynb` | Clean + train |
| `notebooks/03_evaluate_predict.ipynb` | Evaluate + predict |
| `notebooks/04_confusion_matrix.ipynb` | Confusion matrices |

Use the `spotify-genre-prediction` kernel when prompted.

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
| Song lookup warns missing CSV | Put `dataset.csv` in `data/raw/` and refresh the app |
| Upload audio fails on MP3 | Install ffmpeg, or convert/upload a WAV instead |
| `No module named 'librosa'` | `conda install librosa soundfile audioread` (or recreate env from `environment.yml`) |
| `No module named 'streamlit'` | `conda install streamlit` or `pip install streamlit` inside the env |
| `ERR_CONNECTION_REFUSED` in browser | Streamlit isn’t running — start it and keep that terminal open |
| Want to recreate env cleanly | `conda env remove -n spotify-genre-prediction` then `conda env create -f environment.yml` |

---

## License / data credit

- Dataset: [Maharshi Pandya — Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)  
- Follow Kaggle’s and the dataset’s terms when sharing results or derivatives.
