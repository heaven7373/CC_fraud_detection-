"""Create privacy-minimized modeling CSVs from the supplied ZIP archive."""
import csv
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent.parent / "CC_dataset.zip"
OUT = ROOT / "data" / "processed"
KEEP = ["trans_date_trans_time", "category", "amt", "gender", "city_pop", "job", "is_fraud"]

def main():
    archive = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else ARCHIVE
    if not archive.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive}. Pass its path as an argument.")
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for source, target in [("fraudTrain.csv", "fraud_train.csv"), ("fraudTest.csv", "fraud_test.csv")]:
            with zf.open(source) as raw, (OUT / target).open("w", newline="", encoding="utf-8") as output:
                reader = pd.read_csv(raw, usecols=KEEP, chunksize=100_000)
                first = True
                for chunk in reader:
                    chunk["trans_date_trans_time"] = pd.to_datetime(chunk["trans_date_trans_time"], errors="coerce")
                    chunk["amt"] = pd.to_numeric(chunk["amt"], errors="coerce")
                    chunk["city_pop"] = pd.to_numeric(chunk["city_pop"], errors="coerce")
                    chunk["is_fraud"] = pd.to_numeric(chunk["is_fraud"], errors="coerce").astype("Int8")
                    chunk.dropna(subset=KEEP, inplace=True)
                    chunk.to_csv(output, index=False, header=first, date_format="%Y-%m-%d %H:%M:%S")
                    first = False
            print(f"Prepared {source} -> {OUT / target}")

if __name__ == "__main__": main()
