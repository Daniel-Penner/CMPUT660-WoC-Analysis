#First run the following in terminal to create files of sampled blob contents and b2fa relationships:
#zcat ../sample/b2fSampleU.s.gz \ | awk -F';' 'BEGIN{srand(42)} !seen[$1]++ && rand()<0.01 {print $1}' \ > blob_ids.txt
#cat blob_ids.txt | ~/lookup/getValues b2faFullU > blob_first_seen.tsv

#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

HOME = os.path.expanduser("~")

def safe_savefig(name):
    out = os.path.join(HOME, name)
    plt.savefig(out, bbox_inches="tight")
    print(f"[SAVED] {out}")

def plot_counts(df, freq, label):
    counts = df.groupby(pd.Grouper(key="date", freq=freq))["blob"].nunique()
    cumulative_counts = counts.cumsum()

    counts = counts[counts > 0]

    plt.figure(figsize=(12,6))
    counts.plot()
    plt.title(f"Blobs per {label}")
    plt.ylabel("Unique blobs")
    safe_savefig(f"blobs_per_{label.lower()}_line.png")
    plt.close()

    plt.figure(figsize=(12,6))
    cumulative_counts.plot()
    plt.title(f"Cumulative Blobs per {label}")
    plt.ylabel("Cumulative unique blobs")
    safe_savefig(f"blobs_per_{label.lower()}_cumulative.png")
    plt.close()

    return counts, cumulative_counts

def main():
    print("[INFO] Reading blob_first_seen.tsv ...")
    rows = []
    with open("blob_first_seen.tsv", "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            blob, ts = parts[0], parts[1]
            try:
                ts = int(ts)
                rows.append((blob, pd.to_datetime(ts, unit="s")))
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=["blob", "date"])
    df = df[(df["date"].dt.year >= 2005) & (df["date"].dt.year <= 2021)]
    print(f"[INFO] Loaded {len(df)} valid rows from 2005–2021")

    month_counts, month_cum = plot_counts(df, "ME", "Month")


if __name__ == "__main__":
    main()