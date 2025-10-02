#First run the following in terminal to find a list of authors that can be looked up directly for a2p:
# zcat ../A2cSampleU.s.gz | cut -d';' -f1 | sort -u > authors_u.txt
# cat projects_u.txt | ~/lookup/getValues -f a2c > author_commits.tsv

#!/usr/bin/env python3
import os
import sys
import subprocess
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

HOME = os.path.expanduser("~")
LOOKUP_CMD = ["~/lookup/getValues", "-f", "a2p"]

def log(msg):
    print(f"[INFO] {msg}", flush=True)

def safe_savefig(path):
    out = os.path.join(HOME, path)
    plt.savefig(out, bbox_inches="tight")
    print(f"[SAVED] {out}")
    plt.close()

def compute_and_save_stats(values, label, outfile):
    values = np.asarray(values, dtype=float)
    stats = {}
    if values.size == 0:
        stats = {
            "count": 0, "mean": 0, "median": 0, "std": 0, "var": 0,
            "skew": 0, "kurtosis": 0, "q1": 0, "q2": 0, "q3": 0
        }
    else:
        stats = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "var": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
            "skew": float(skew(values, bias=False)) if values.size > 1 else 0.0,
            "kurtosis": float(kurtosis(values, bias=False)) if values.size > 1 else 0.0,
            "q1": float(np.percentile(values, 25)),
            "q2": float(np.percentile(values, 50)),
            "q3": float(np.percentile(values, 75)),
        }
    outpath = os.path.join(HOME, outfile)
    with open(outpath, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    log(f"{label} stats written to {outpath}")
    return stats

def read_sampled_authors(tsv_path, step):
    seen = set()
    sampled = []
    with open(tsv_path, "r", errors="ignore") as f:
        for idx, line in enumerate(f):
            if idx % step != 0:
                continue
            parts = line.strip().split(";")
            if not parts:
                continue
            author = parts[0]
            if author and author not in seen:
                seen.add(author)
                sampled.append(author)
    log(f"From {tsv_path}: sampled {len(sampled)} unique authors (every 1/{step} rows).")
    return sampled

def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def lookup_projects_for_authors(authors):
    a2p_map = defaultdict(set)
    total = len(authors)
    processed = 0
    for batch in batched(authors, 2000):
        proc = subprocess.Popen(
            " ".join(LOOKUP_CMD),
            shell=True,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out, _ = proc.communicate("\n".join(batch))
        for line in out.strip().splitlines():
            parts = line.split(";")
            if len(parts) >= 2:
                author, project = parts[0], parts[1]
                a2p_map[author].add(project)
        processed += len(batch)
        log(f"lookup a2p: processed {processed}/{total} authors...")
    return a2p_map

def make_boxplot(values, stem):
    plt.figure(figsize=(8,6))
    plt.boxplot(values, vert=True, showfliers=False, labels=["Projects per Author"])
    plt.title("Projects per Author (Linear)")
    plt.ylabel("Projects")
    safe_savefig(f"{stem}_box_linear.png")

def make_cdf(values, stem):
    sorted_vals = np.sort(values)
    y = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

    plt.figure(figsize=(10,6))
    plt.plot(sorted_vals, y, marker=".", linestyle="none")
    plt.xscale("log")
    plt.title("CDF of Projects per Author (Log X)")
    plt.xlabel("Projects (log)")
    plt.ylabel("Cumulative Probability")
    safe_savefig(f"{stem}_cdf_logx.png")

def main():

    authors = read_sampled_authors("author_commits.tsv", 100)
    if not authors:
        log("No sampled authors found. Exiting.")
        return

    a2p_map = lookup_projects_for_authors(authors)
    counts = [len(v) for v in a2p_map.values() if v]
    log(f"Authors with projects: {len(counts)} / sampled {len(authors)}")

    if not counts:
        log("No projects found for sampled authors. Try a different step or input.")
        return

    compute_and_save_stats(counts, "Projects per Author",
                           "overlap_projects_per_author_stats.txt")
    make_boxplot(counts, "overlap_projects_per_author")
    make_cdf(counts, "overlap_projects_per_author")
    log("Done.")

if __name__ == "__main__":
    main()