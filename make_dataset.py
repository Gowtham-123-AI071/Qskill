# backend/src/data/make_dataset.py
import os
from pathlib import Path
import requests
import zipfile
import io
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

def download_and_extract():
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(RAW_DIR)
    # raw file is at RAW_DIR/SMSSpamCollection
    raw_file = RAW_DIR / "SMSSpamCollection"
    df = pd.read_csv(raw_file, sep="\t", header=None, names=["label", "text"])
    # normalize labels to 'spam'/'ham'
    df['label'] = df['label'].str.strip()
    df.to_csv(RAW_DIR / "sms_spam_raw.csv", index=False)
    print("Saved raw csv to", RAW_DIR / "sms_spam_raw.csv")

if __name__ == "__main__":
    download_and_extract()
