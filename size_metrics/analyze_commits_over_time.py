#!/usr/bin/env python3
import os
import gzip
import pandas as pd
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")

def log(msg):
    print(f"[INFO] {msg}", flush=True)

def safe_savefig(name):
    out = os.path.join(HOME, name)
    plt.savefig(out, bbox_inches="tight")
    log(f"[SAVED] {out}")
    plt.close()

def load_commits_from_sample():
    timestamps = []
    path = "../sampling/sample/c2datSampleU.s.gz"
    if not os.path.exists(path):
        log(f"Sample file not found: {path}")
        return []

    log(f"Reading {path} ...")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            try:
                ts = int(parts[1])
                timestamps.append(ts)
            except ValueError:
                continue

    log(f"Loaded {len(timestamps)} commit timestamps total.")
    return timestamps

def plot_over_time(timestamps):
    df = pd.DataFrame({"ts": timestamps})
    df["date"] = pd.to_datetime(df["ts"], unit="s")

    # Filter 2005–2021
    df = df[(df["date"].dt.year >= 2005) & (df["date"].dt.year <= 2021)]
    log(f"Filtered to {len(df)} commits between 2005–2021.")

    if df.empty:
        log("No commits in range — skipping plots.")
        return

    monthly_counts = df.groupby(pd.Grouper(key="date", freq="ME")).size()
    monthly_cumulative = monthly_counts.cumsum()

    plt.figure(figsize=(12,6))
    monthly_counts.plot(title="Commits per Month (2005–2021)")
    plt.ylabel("Commits")
    safe_savefig("commits_per_month.png")

    plt.figure(figsize=(12,6))
    monthly_cumulative.plot(title="Cumulative Commits per Month (2005–2021)")
    plt.ylabel("Cumulative Commits")
    safe_savefig("commits_per_month_cumulative.png")

def main():
    timestamps = load_commits_from_sample()
    if not timestamps:
        log("No commit timestamps found. Exiting.")
        return
    plot_over_time(timestamps)
    log("Done.")

if __name__ == "__main__":
    main()