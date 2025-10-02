#First run the following in terminal to create a smaller sample of just blob ids:
#zcat ../sample/b2fSampleU.s.gz \ | awk -F';' 'BEGIN{srand(42)} !seen[$1]++ && rand()<0.001 {print $1}' \ > blob_ids.txt

#!/usr/bin/env python3
import subprocess
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

BLOB_FILE = "blob_ids.txt"   # default input file

def get_blob_size(blob_id: str):
    #Fetch blob contents from WoC and return its size in bytes.
    cmd = f"echo {blob_id.strip()} | ~/lookup/showCnt blob 1"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True).strip()
        if ";" not in output or output.startswith("no blob"):
            return None 
        b64 = output.split(";", 1)[1]
        return len(base64.b64decode(b64))
    except Exception as e:
        print(f"[WARN] Failed to get blob {blob_id}: {e}")
        return None

def main():
    blob_sizes = []
    skipped = 0
    start = time.time()

    with open(BLOB_FILE, "r") as f:
        for idx, line in enumerate(f, 1):
            blob_id = line.strip()
            size = get_blob_size(blob_id)

            if size is None:
                skipped += 1
                continue

            blob_sizes.append(size)

            if len(blob_sizes) % 100 == 0:
                elapsed = time.time() - start
                avg = np.mean(blob_sizes)
                print(f"[INFO] Processed {len(blob_sizes)} valid blobs "
                      f"(line {idx}) in {elapsed:.1f}s | Current mean size: {avg:.2f} bytes")

    blob_sizes = np.array(blob_sizes)
    n = len(blob_sizes)
    print("\n===== RESULTS =====")
    print(f"Blobs processed (valid): {n}")
    print(f"Blobs skipped (missing): {skipped}")
    print(f"Total size (sample): {blob_sizes.sum():.0f} bytes")
    print(f"Mean: {blob_sizes.mean():.2f} bytes")
    print(f"Median: {np.median(blob_sizes):.2f} bytes")
    print(f"Variance: {blob_sizes.var():.2f}")
    print(f"Skewness: {skew(blob_sizes):.2f}")
    print(f"Kurtosis: {kurtosis(blob_sizes):.2f}")
    print("Quartiles (Q1, Q2, Q3):", np.percentile(blob_sizes, [25, 50, 75]))

    plt.figure(figsize=(10, 6))
    plt.boxplot(blob_sizes, vert=False, showfliers=False, labels=["Blob Sizes"])
    plt.title("Blob Sizes - Boxplot (Linear Scale)")
    plt.xlabel("Size (bytes)")
    plt.savefig("blob_sizes_boxplot_linear.png")

    plt.figure(figsize=(10, 6))
    sorted_sizes = np.sort(blob_sizes)
    cdf = np.arange(1, n+1) / n
    plt.plot(sorted_sizes, cdf, marker=".", linestyle="none")
    plt.xscale("log") 
    plt.title("Blob Sizes - CDF")
    plt.xlabel("Size (bytes, log scale)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.savefig("blob_sizes_cdf.png")

if __name__ == "__main__":
    main()
