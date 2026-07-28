# Spotify Genre Prediction

## What this project is

This project asks:

> **How accurately can a supervised machine learning model predict a song's genre from its Spotify audio features alone?**

We use the public [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~114,000 tracks, many genres, and audio features such as danceability, energy, loudness, tempo, etc.).

**Current status:**

1. Build a reproducible conda environment from `environment.yml`
2. Download / load the Kaggle Spotify tracks dataset
3. Clean data, collapse similar genres, train a model, and save it
4. Evaluate the held-out test set and save predictions
5. Create visual confusion matrices
6. Run a Streamlit app (`app.py`) to predict genre from audio features

**Note on “input a song”:** the model predicts from **audio features** (danceability, energy, tempo, …), not from a song title alone. The app lets you set those features with sliders or load a test-set example. Looking up a song by name would require an external music API (future enhancement).

---

## What is a “repo structure”?

**Yes — the repo structure is just the folders and files in this GitHub repository** (and in your local/OSC copy of the project).

It is a map of *where things live*, so anyone who clones the repo knows:

- where to put data
- which scripts to run
- which notebook to open
- which file creates the software environment

It is **not** a separate thing from GitHub. When you push this project to GitHub, that same tree of files *is* the repository.

### Current repository structure

```text
spotify-genre-prediction/
├── README.md
├── environment.yml
├── app.py                    # Streamlit UI — predict genre from audio features
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── raw/                  # dataset.csv (not committed)
│   └── processed/            # cleaned tables (local)
│
├── notebooks/
│   ├── 01_load_data.ipynb
│   ├── 02_train_model.ipynb
│   ├── 03_evaluate_predict.ipynb
│   └── 04_confusion_matrix.ipynb
│
├── scripts/
│   ├── download_data.py
│   ├── load_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── plot_confusion_matrix.py
│
├── src/
│   ├── data_io.py
│   ├── clean.py
│   ├── genre_map.py
│   ├── train.py
│   ├── evaluate.py
│   └── plots.py
│
├── models/                   # genre_classifier.joblib + metrics
└── outputs/                  # predictions CSV + confusion matrix PNGs
```

**Important:** `data/raw/dataset.csv` is **not** committed to GitHub (it is large and comes from Kaggle). Everyone downloads it themselves after cloning.

---

## Who this README is for

You can run this project on:

- **OSC OnDemand** (Ohio Supercomputer Center), or
- Your own laptop/desktop with **conda** / **miniconda** / **anaconda**

The steps are the same idea everywhere: create the env → get the data → run scripts/notebook.

---

## Prerequisites

Before you start:

1. Access to a terminal (OSC shell app, or Terminal on a Mac)
2. Conda available (on OSC: load a miniconda module; on a laptop: install Miniconda/Anaconda)
3. A copy of this project folder on that machine
4. A Kaggle account (free) if you want to download the dataset from Kaggle

---

## Step 0 — Get the project onto the machine you will use

### Option A — Clone from GitHub (after the repo is on GitHub)

```bash
cd ~
git clone <YOUR_GITHUB_REPO_URL>.git
cd spotify-genre-prediction
ls
```

You should see `environment.yml`, `scripts/`, `notebooks/`, `src/`, and `data/`.

### Option B — Upload from your Mac to OSC (useful before the first GitHub push)

1. On your Mac, zip the project (or use the upload zip you already made).
2. In OSC OnDemand → **Files** → **Home Directory** → **Upload** the zip.
3. In the OSC terminal:

```bash
cd ~
unzip spotify-genre-prediction-osc-upload.zip
# or whatever the zip file is named
cd spotify-genre-prediction
ls
```

### Confirm you are in the project root

```bash
pwd
ls
```

You want to be inside `.../spotify-genre-prediction` and see `environment.yml` listed.

If you run scripts from your home folder (`~`) instead of the project folder, you will get errors like:

`python: can't open file '.../scripts/download_data.py'`

---

## Step 1 — Create the conda environment from `environment.yml`

`environment.yml` is the recipe for a reproducible environment. Anyone who uses this file should get the same core packages (Python, pandas, scikit-learn, Jupyter, etc.).

### On OSC OnDemand

```bash
# 1) Go to the project
cd ~/spotify-genre-prediction

# 2) Make conda available (module name may vary slightly on OSC)
module load miniconda3/24.1.2-py310

# 3) Create the environment from the file
conda env create -f environment.yml

# 4) Activate it
conda activate spotify-genre-prediction

# 5) Confirm it worked
conda info --envs
python --version
which python
```

### On a personal laptop (conda already installed)

```bash
cd /path/to/spotify-genre-prediction
conda env create -f environment.yml
conda activate spotify-genre-prediction
python --version
```

### If conda says the environment name already exists

```bash
# Either use a new name:
conda env create -f environment.yml -n spotify-genre-prediction

# Or update the existing env:
conda env update -n spotify-genre-prediction -f environment.yml --prune

# Or remove and recreate (destructive for that env only):
conda env remove -n spotify-genre-prediction
conda env create -f environment.yml
```

### If `conda: command not found` on OSC

You forgot to load the module:

```bash
module load miniconda3/24.1.2-py310
# then retry conda commands
```

### Optional: install/repair the Kaggle package inside the env

If download later says `No module named 'kaggle'`:

```bash
conda activate spotify-genre-prediction
pip install kaggle
```

---

## Step 2 — Get the dataset into `data/raw/`

Expected file after setup:

```text
data/raw/dataset.csv
```

Create the folder if it is missing:

```bash
cd ~/spotify-genre-prediction
mkdir -p data/raw data/processed
```

### Method A — Manual download (simplest; recommended first)

1. Open the dataset page while logged into Kaggle:  
   https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Accept any dataset terms if prompted.
3. Click **Download** to get a ZIP file.
4. Put the ZIP **or** the extracted `dataset.csv` here:

```text
spotify-genre-prediction/data/raw/
```

   On OSC OnDemand: **Files** → `spotify-genre-prediction` → `data` → `raw` → **Upload**.

5. Then run the download script (it will unzip if needed):

```bash
cd ~/spotify-genre-prediction
conda activate spotify-genre-prediction
python scripts/download_data.py
```

### Method B — Automatic download with the Kaggle API

1. On Kaggle: **Settings → API → Create New Token**  
   This downloads a file named `kaggle.json`.
2. On the machine where you run the project:

```bash
mkdir -p ~/.kaggle
# copy kaggle.json into ~/.kaggle/ (upload via OnDemand Files if on OSC)
chmod 600 ~/.kaggle/kaggle.json
```

3. Open the dataset page once in a browser and accept terms.
4. Install the client if needed, then download:

```bash
conda activate spotify-genre-prediction
pip install kaggle
cd ~/spotify-genre-prediction
python scripts/download_data.py
```

### Confirm the data file exists

```bash
ls -lh data/raw/dataset.csv
```

You should see a CSV roughly around **~20 MB**.

---

## Step 3 — Run the scripts

Always run these from the **project root** (`spotify-genre-prediction/`), with the env activated.

### Download / unzip helper

```bash
cd ~/spotify-genre-prediction
conda activate spotify-genre-prediction
python scripts/download_data.py
```

Force a fresh download attempt (if using the API):

```bash
python scripts/download_data.py --force
```

### Load and validate

```bash
python scripts/load_data.py
```

If the CSV is missing, this script can try to download first:

```bash
python scripts/load_data.py --download
```

### What success looks like in the terminal

From `download_data.py`, something like:

```text
Dataset already present: .../data/raw/dataset.csv
Done. CSV path: .../data/raw/dataset.csv
```

or unzip/download messages ending in `Ready:` / `Done.`

From `load_data.py`, something like:

```text
Shape: 114,000 rows x ... columns
Columns (...): [..., 'danceability', 'energy', ..., 'track_genre', ...]
Genres (track_genre): 125 unique
Top 10 genres by track count:
  ...
All expected audio feature columns are present.

First 5 rows:
  ...
```

Exact row/column counts can vary slightly by dataset version, but you should **not** see a Python traceback at the end.

---

## Step 4 — Run the notebook

### Start Jupyter (OSC or laptop)

```bash
cd ~/spotify-genre-prediction
conda activate spotify-genre-prediction
jupyter notebook
```

On OSC OnDemand you can also use a **Jupyter** app/job if your center provides one. In that case:

1. Open the Jupyter file browser
2. Navigate into `spotify-genre-prediction/notebooks/`
3. Open `01_load_data.ipynb`

### Inside the notebook

1. Select the `spotify-genre-prediction` kernel if prompted (or the Python from that conda env)
2. Run all cells **in order** (top to bottom)

The notebook will:

- locate the project root
- download/unzip if needed
- load `dataset.csv`
- print/validate shape, genres, and audio feature columns

### Checkpoint 1 success checklist

- [ ] `conda activate spotify-genre-prediction` works
- [ ] `data/raw/dataset.csv` exists
- [ ] `python scripts/load_data.py` prints shape + genre summary with no error
- [ ] `notebooks/01_load_data.ipynb` runs cleanly end-to-end

---

## Step 5 — Run the Streamlit app (Genre Signal)

The app loads the trained model and lets you predict a **parent genre** from audio-feature controls. It can also show test metrics and confusion matrices.

### Install Streamlit if needed

If your env was created before Streamlit was added to `environment.yml`:

```bash
conda activate spotify-genre-prediction
conda install -y streamlit
# or: pip install streamlit
```

### Launch (laptop)

```bash
cd /path/to/SU26-7030-Capstone-   # or spotify-genre-prediction
conda activate spotify-genre-prediction
streamlit run app.py
```

Your browser should open a local page (usually `http://localhost:8501`).

### Launch notes for OSC

Streamlit needs a browser connection to the process. On OSC, easiest options are:

1. Run Streamlit on a machine where you can open a browser to that host/port, or  
2. Use SSH local port forwarding from your laptop, for example:

```bash
# on OSC
cd ~/SU26-7030-Capstone-
conda activate spotify-genre-prediction
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

Then from your laptop (in a separate terminal), forward the port if your OSC setup supports SSH tunneling, and open `http://localhost:8501`.

If port forwarding is awkward on your account, you can still develop/demo the app on your Mac after `git pull`.

### What the app can do

- Adjust Spotify-style audio features with sliders  
- Click **Predict genre** to see the label + confidence + top probabilities  
- Optionally load a row from `outputs/test_predictions.csv`  
- View saved test metrics and confusion matrix images  

**Limitation:** entering only a song title is not supported yet (needs an external music API to fetch audio features).

---

## What the data columns mean (high level)

The label we want to predict later is:

- **`track_genre`** — genre label for the track

Audio features used for prediction (examples):

- **`danceability`**, **`energy`**, **`loudness`**, **`tempo`**, **`valence`**
- **`acousticness`**, **`instrumentalness`**, **`speechiness`**, **`liveness`**
- **`key`**, **`mode`**, **`time_signature`**, **`duration_ms`**

Other columns (track name, artists, etc.) are useful for display in a future app, but the modeling question focuses on **audio features → genre**.

---

## Common problems and fixes

| Problem | Likely cause | Fix |
|--------|---------------|-----|
| `conda: command not found` | Conda module not loaded (OSC) | `module load miniconda3/24.1.2-py310` |
| `prefix already exists: .../envs/ai` | Different/old `environment.yml` or env name conflict | Use `-n spotify-genre-prediction` or remove/update that env |
| `can't open file '.../scripts/download_data.py'` | Wrong current directory | `cd ~/spotify-genre-prediction` then rerun |
| `No such file .../data/raw` | Folder not created yet | `mkdir -p data/raw` |
| `No module named 'kaggle'` | Package missing in env | `pip install kaggle` (or use manual ZIP upload) |
| `Could not find dataset.csv` | Data not uploaded / download failed | Put ZIP or CSV in `data/raw/`, rerun `download_data.py` |
| `No module named 'streamlit'` | Streamlit not installed in env | `conda install streamlit` or `pip install streamlit` |
| Streamlit opens but model missing | `models/genre_classifier.joblib` not present | `git pull` or retrain with `python scripts/train_model.py` |

---

## Reproducing the environment on another machine

Someone who only wants to look at and run the project should:

1. Clone the GitHub repository
2. Create the env: `conda env create -f environment.yml`
3. Activate: `conda activate spotify-genre-prediction`
4. Follow **Step 2** to obtain `data/raw/dataset.csv`
5. Run `python scripts/load_data.py` and/or open `notebooks/01_load_data.ipynb`

That is the purpose of `environment.yml` + this README: **same setup instructions everywhere** (OSC or laptop).

---

## Project status / roadmap

| Piece | Status |
|------|--------|
| `environment.yml` | Done |
| Data download + load scripts | Done |
| Train / clean / collapse genres + saved model | Done |
| Test evaluation + predictions CSV | Done |
| Confusion matrix figures | Done |
| Streamlit app (`app.py`) | Done |
| Song-title / Spotify API lookup | Optional future |

---

## License / data credit

- Dataset: [Maharshi Pandya — Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- Please follow Kaggle’s and the dataset’s terms when redistributing results or derivatives.
