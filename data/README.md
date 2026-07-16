# Spotify Tracks dataset (raw)

Source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Expected file after download: `dataset.csv`

## Automatic download (Kaggle API)

1. Create a Kaggle account and open the dataset page above (accept terms if asked).
2. Kaggle → Settings → API → Create New Token → save as `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`
4. From the project root (env activated):

```bash
python scripts/download_data.py
python scripts/load_data.py
```

## Manual download

1. Download the ZIP from the Kaggle page.
2. Place the ZIP or `dataset.csv` in this folder (`data/raw/`).
3. Run:

```bash
python scripts/download_data.py   # will unzip if needed
python scripts/load_data.py
```
