#First run the following in terminal to find a list of projects that can be looked up directly for p2c:
# zcat ../c2PSampleU.s.gz | cut -d';' -f2 | sort -u > projects_u.txt
# cat projects_u.txt | ~/lookup/getValues -f p2c > project_commits.tsv

#!/usr/bin/env python3
import os
import subprocess
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

HOME = os.path.expanduser("~")
LOOKUP_CMD = ["~/lookup/getValues", "-f", "p2c"]

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

def read_sampled_projects(tsv_path, step):
    """Deterministically sample every 1/step project from the TSV's first column."""
    seen = set()
    sampled = []
    with open(tsv_path, "r", errors="ignore") as f:
        for idx, line in enumerate(f):
            if idx % step != 0:
                continue
            parts = line.strip().split(";")
            if not parts:
                continue
            proj = parts[0]
            if proj and proj not in seen:
                seen.add(proj)
                sampled.append(proj)
    log(f"From {tsv_path}: sampled {len(sampled)} unique projects (every 1/{step} rows).")
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

def lookup_commits_for_projects(projects):
    """Use lookup (V) p2c to get all commits for each project."""
    proj_to_commits = defaultdict(list)
    total = len(projects)
    processed = 0
    for batch in batched(projects, 2000):
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
                proj, commit = parts[0], parts[1]
                proj_to_commits[proj].append(commit)
        processed += len(batch)
        log(f"lookup p2c: processed {processed}/{total} projects...")
    return proj_to_commits

def make_boxplot(values, stem):
    plt.figure(figsize=(8,6))
    plt.boxplot(values, vert=True, showfliers=False, labels=["Commits per Project"])
    plt.title("Commits per Project (Linear)")
    plt.ylabel("Commits")
    safe_savefig(f"{stem}_box_linear.png")

def make_cdf(values, stem):
    sorted_vals = np.sort(values)
    y = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

    plt.figure(figsize=(10,6))
    plt.plot(sorted_vals, y, marker=".", linestyle="none")
    plt.xscale("log")
    plt.title("CDF of Commits per Project (Log X)")
    plt.xlabel("Commits (log)")
    plt.ylabel("Cumulative Probability")
    safe_savefig(f"{stem}_cdf_logx.png")

def main():

    projects = read_sampled_projects("project_commits.tsv", 100)
    if not projects:
        log("No sampled projects found. Exiting.")
        return

    proj_to_commits = lookup_commits_for_projects(projects)
    # Keep only projects that returned commits
    counts = [len(v) for v in proj_to_commits.values() if v]
    log(f"Projects with commits: {len(counts)} / sampled {len(projects)}")

    if not counts:
        log("No commits found for sampled projects. Try a different step or input.")
        return

    compute_and_save_stats(counts, "Commits per Project",
                           "overlap_commits_per_project_stats.txt")
    make_boxplot(counts, "overlap_commits_per_project")
    make_cdf(counts, "overlap_commits_per_project")
    log("Done.")

if __name__ == "__main__":
    main()