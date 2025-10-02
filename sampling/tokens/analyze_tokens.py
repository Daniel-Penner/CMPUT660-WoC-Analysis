#First run the following in terminal to create a smaller sample of just blob ids:
#zcat ../sample/b2fSampleU.s.gz \ | awk -F';' 'BEGIN{srand(42)} !seen[$1]++ && rand()<0.001 {print $1}' \ > blob_ids.txt

#!/usr/bin/env python3
import os
import re
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

HOME = os.path.expanduser("~")
SAMPLE_SIZE = 10000 #not all 10,000 will be found. Actual sampled amount shown in output
TOTAL_BLOBS = 12490439543 #From WoC website

def log(msg):
    print(f"[INFO] {msg}", flush=True)

def safe_savefig(name):
    out = os.path.join(HOME, name)
    plt.savefig(out, bbox_inches="tight")
    log(f"[SAVED] {out}")
    plt.close()

def tokenize(text):
    tokens = re.split(r"\W+", text)
    return [t for t in tokens if t]

def get_blob_content(blob_id):
    """Fetch blob content using showCnt (handle binary safely)."""
    cmd = ["~/lookup/showCnt", "blob"]
    proc = subprocess.Popen(
        " ".join(cmd),
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, _ = proc.communicate((blob_id + "\n").encode("utf-8"))
    try:
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------- Heaps’ Law ----------
def heaps_law(N, K, beta):
    return K * (N ** beta)

def fit_heaps(growth_points):
    pts = [(x, y) for x, y in growth_points if x > 0 and y > 0]
    if len(pts) < 3:
        return 0.0, 0.0
    xs, ys = zip(*pts)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    popt, _ = curve_fit(heaps_law, xs, ys, bounds=([0,0], [1e12, 1]))
    K, beta = popt
    log(f"Heaps’ Law fit: K={K:.4g}, beta={beta:.4f}")
    return K, beta

def analyze_tokens(blob_ids):
    total_tokens_per_blob = []
    unique_tokens_per_blob = []
    global_token_set = set()
    global_total_tokens = 0

    growth_points = []  # track vocab growth for Heaps’ Law

    for idx, blob in enumerate(blob_ids, 1):
        raw = get_blob_content(blob)
        toks = tokenize(raw)
        if not toks:
            continue
        total_tokens_per_blob.append(len(toks))
        unique_tokens_per_blob.append(len(set(toks)))
        global_total_tokens += len(toks)
        global_token_set.update(toks)

        # record growth
        growth_points.append((global_total_tokens, len(global_token_set)))

        if idx % 50 == 0:
            log(f"Processed {idx}/{len(blob_ids)} blobs...")

    return total_tokens_per_blob, unique_tokens_per_blob, global_total_tokens, len(global_token_set), growth_points

def compute_stats(values, label, fname):
    values = np.asarray(values)
    stats = {
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else 0,
        "median": float(np.median(values)) if values.size else 0,
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0,
        "var": float(np.var(values, ddof=1)) if values.size > 1 else 0,
        "skew": float(skew(values, bias=False)) if values.size > 1 else 0,
        "kurtosis": float(kurtosis(values, bias=False)) if values.size > 1 else 0,
        "q1": float(np.percentile(values, 25)) if values.size else 0,
        "q2": float(np.percentile(values, 50)) if values.size else 0,
        "q3": float(np.percentile(values, 75)) if values.size else 0,
        "min": float(np.min(values)) if values.size else 0,
        "max": float(np.max(values)) if values.size else 0,
    }
    outpath = os.path.join(HOME, fname)
    with open(outpath, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    log(f"{label} stats written to {outpath}")
    return stats

def make_boxplot(values, label, fname, logscale=False):
    plt.figure(figsize=(8,6))
    plt.boxplot(values, vert=True, showfliers=False, labels=[label])
    plt.title(f"{label} {'(Log)' if logscale else '(Linear)'}")
    plt.ylabel(label)
    if logscale:
        plt.yscale("log")
    safe_savefig(fname)

def make_cdf(values, label, fname, logx=False):
    sorted_vals = np.sort(values)
    y = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    plt.figure(figsize=(10,6))
    plt.plot(sorted_vals, y, marker=".", linestyle="none")
    plt.title(f"CDF of {label} {'(Log X)' if logx else '(Linear X)'}")
    plt.xlabel(label)
    plt.ylabel("Cumulative Probability")
    if logx:
        plt.xscale("log")
    safe_savefig(fname)

def main():
    with open("blob_ids.txt") as f:
        all_blobs = [line.strip() for line in f if line.strip()]
    sample = random.sample(all_blobs, min(SAMPLE_SIZE, len(all_blobs)))
    log(f"Loaded {len(sample)} blob IDs to process.")

    totals, uniques, global_total, global_unique, growth_points = analyze_tokens(sample)

    log(f"Global totals across sample:")
    log(f"  Total tokens = {global_total}")
    log(f"  Unique tokens = {global_unique}")

    K, beta = fit_heaps(growth_points)
    est_total = int(heaps_law(TOTAL_BLOBS, K, beta)) if K > 0 else 0
    with open(os.path.join(HOME, "heaps_law_summary.txt"), "w") as f:
        f.write(f"Blobs sampled: {len(sample)}\n")
        f.write(f"Total tokens in sample: {global_total}\n")
        f.write(f"Unique tokens in sample: {global_unique}\n")
        f.write(f"Heaps K={K:.6g}, beta={beta:.6f}\n")
        f.write(f"TOTAL_BLOBS: {TOTAL_BLOBS}\n")
        f.write(f"Heaps projected unique tokens: {est_total}\n")
    log("[STATS] heaps_law_summary.txt written")

    compute_stats(totals, "Tokens per Blob", "tokens_per_blob_stats.txt")
    compute_stats(uniques, "Unique Tokens per Blob", "unique_tokens_per_blob_stats.txt")

    make_boxplot(totals, "Tokens per Blob", "tokens_per_blob_box_linear.png")
    make_boxplot(totals, "Tokens per Blob", "tokens_per_blob_box_log.png", logscale=True)
    make_cdf(totals, "Tokens per Blob", "tokens_per_blob_cdf_linear.png")
    make_cdf(totals, "Tokens per Blob", "tokens_per_blob_cdf_log.png", logx=True)

    make_boxplot(uniques, "Unique Tokens per Blob", "unique_tokens_per_blob_box_linear.png")
    make_boxplot(uniques, "Unique Tokens per Blob", "unique_tokens_per_blob_box_log.png", logscale=True)
    make_cdf(uniques, "Unique Tokens per Blob", "unique_tokens_per_blob_cdf_linear.png")
    make_cdf(uniques, "Unique Tokens per Blob", "unique_tokens_per_blob_cdf_log.png", logx=True)

    with open(os.path.join(HOME, "token_global_summary.txt"), "w") as f:
        f.write(f"Blobs sampled: {len(sample)}\n")
        f.write(f"Total tokens across sample: {global_total}\n")
        f.write(f"Unique tokens across sample: {global_unique}\n")

    log("Done.")

if __name__ == "__main__":
    main()