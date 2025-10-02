#First run the following in terminal to create files of sampled blob contents and b2f relationships:
#zcat ../sample/b2fSampleU.s.gz \ | awk -F';' 'BEGIN{srand(42)} !seen[$1]++ && rand()<0.01 {print $1}' \ > blob_ids.txt
# cat blob_ids.txt | ~/lookup/showCnt blob 1 > blobs_sample_content.txt
# cat blob_ids.txt | ~/lookup/getValues -f b2f > blob_files.tsv

#!/usr/bin/env python3
import os, re, sys, base64, subprocess
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
BLOB_CONTENT_FILE = "blobs_sample_content.txt"
BLOB_FILES_TSV    = "blob_files.tsv" 
OUTDIR = os.path.expanduser("~") 
TOP_K_ENTITIES    = 8
BATCH             = 5000

GETVALUES = os.path.expanduser("~/lookup/getValues")

# Internal (not foreign) URL domains
INTERNAL_DOMAINS = {
    "github.com","gist.github.com","raw.githubusercontent.com","api.github.com",
    "gitlab.com","bitbucket.org"
}

# Regex for URLs
URL_RE = re.compile(r"https?://[^\s)\"'>]+")

def parse_blobs_one_line(path):
    """Read blobs_sample_content.txt and decode base64 into text blobs."""
    blobs_text = {}
    total, textlike = 0, 0
    with open(path, "r") as f:
        for line in f:
            total += 1
            parts = line.rstrip("\n").split(";", 1)
            if len(parts) != 2:
                continue
            blob, b64 = parts
            try:
                raw = base64.b64decode(b64, validate=False)
            except Exception:
                continue
            if b"\x00" in raw:  # binary check
                continue
            text = raw.decode("utf-8", errors="ignore")
            if not text:
                continue
            printable = sum(1 for ch in text if (ch >= " " or ch in "\n\r\t"))
            if printable / max(1, len(text)) < 0.95:
                continue
            if len(raw) > 1_000_000:
                continue
            blobs_text[blob] = text
            textlike += 1
    return blobs_text, total, textlike

def extract_urls(text):
    return URL_RE.findall(text)

def domain_of(url):
    try:
        return url.split("/")[2].lower()
    except Exception:
        return None

def run_getvalues(map_name, keys):
    if not keys:
        return []
    p = subprocess.Popen([GETVALUES, map_name],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdin_data = ("\n".join(keys) + "\n").encode("utf-8")
    out, err = p.communicate(stdin_data)
    if p.returncode != 0 and err:
        sys.stderr.buffer.write(err)
    # decode safely
    out_text = out.decode("utf-8", errors="ignore")
    return [ln for ln in out_text.splitlines() if ln.strip()]


def parse_b2tac_lines(lines):
    blob_year = {}
    for ln in lines:
        parts = ln.split(";")
        if len(parts) < 2:
            continue
        blob = parts[0]
        ts = None
        for i in range(1, len(parts)):
            try:
                ts = int(parts[i])
                break
            except Exception:
                continue
        if ts is None:
            continue
        year = pd.to_datetime(ts, unit="s").year
        blob_year[blob] = year
    return blob_year


def parse_blob_files(blob_files_tsv):
    ext_to_lang = {
        ".py":"Python",".ipynb":"Jupyter",".js":"JavaScript",".ts":"TypeScript",
        ".tsx":"TypeScript",".jsx":"JavaScript",".java":"Java",".c":"C",".h":"C",
        ".cpp":"C++",".cc":"C++",".cxx":"C++",".hpp":"C++",".hh":"C++",
        ".rb":"Ruby",".go":"Go",".php":"PHP",".rs":"Rust",".kt":"Kotlin",
        ".swift":"Swift",".m":"Objective-C",".scala":"Scala",".cs":"C#",
        ".sh":"Shell",".bash":"Shell",".zsh":"Shell",".ps1":"PowerShell",
        ".json":"JSON",".yml":"YAML",".yaml":"YAML",".xml":"XML",".toml":"TOML",
        ".ini":"INI",".cfg":"CFG",".md":"Markdown",".rst":"reStructuredText",
        ".txt":"Text",".html":"HTML",".css":"CSS",".sql":"SQL",".pl":"Perl",
        ".lua":"Lua",".r":"R",".dart":"Dart"
    }
    blob_to_langs = defaultdict(set)
    with open(blob_files_tsv, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ";" not in ln:
                continue
            blob, path = ln.rstrip("\n").split(";", 1)
            _, ext = os.path.splitext(os.path.basename(path))
            if ext in ext_to_lang:
                blob_to_langs[blob].add(ext_to_lang[ext])
    return blob_to_langs

def classify_script_mix(text):
    """Heuristic: detect Latin/Cyrillic/CJK mix in text blobs."""
    scripts = {"latin":0,"cyr":0,"cjk":0}
    for ch in text:
        o = ord(ch)
        if (0x0041 <= o <= 0x024F) or (0x1E00 <= o <= 0x1EFF):
            scripts["latin"] += 1
        elif (0x0400 <= o <= 0x052F):
            scripts["cyr"] += 1
        elif (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF):
            scripts["cjk"] += 1
    return {s for s,c in scripts.items() if c >= 20}

def main():

    blobs_text, total_sampled, textlike = parse_blobs_one_line(BLOB_CONTENT_FILE)
    print(f"[INFO] Blob content lines: {total_sampled:,}")
    print(f"[INFO] Text-like blobs:    {textlike:,}")

    blob_has_url = set()
    url_domains_blob = Counter()
    url_sources = []
    for blob, text in blobs_text.items():
        for url in extract_urls(text):
            dom = domain_of(url)
            if not dom:
                continue
            blob_has_url.add(blob)
            url_domains_blob[dom] += 1
            url_sources.append(("blob_text", dom, blob, None))


    total_urls = len(url_sources)
    foreign_urls = sum(1 for (_, dom, _, _) in url_sources if dom not in INTERNAL_DOMAINS)

    print(f"[INFO] Total URLs (raw): {total_urls:,}")
    print(f"[INFO] Foreign URLs (raw): {foreign_urls:,} ({foreign_urls/total_urls:.2%})")

    with open(os.path.join(OUTDIR, "foreign_url_totals.txt"), "w") as f:
        f.write(f"Total URLs: {total_urls}\n")
        f.write(f"Foreign URLs: {foreign_urls}\n")
        f.write(f"Percent foreign: {foreign_urls/total_urls:.2%}\n")

    blob_to_year = {}
    blobs_needing_time = list(blob_has_url)
    print(f"[INFO] Querying b2tac for {len(blobs_needing_time):,} blobs...")
    for i in range(0, len(blobs_needing_time), BATCH):
        chunk = blobs_needing_time[i:i+BATCH]
        lines = run_getvalues("b2tac", chunk)
        blob_to_year.update(parse_b2tac_lines(lines))

    rows_with_time = []
    for source, dom, ident, yr in url_sources:
        if source == "blob_text":
            yr2 = blob_to_year.get(ident)
            if yr2 is None:
                continue
            rows_with_time.append((source, dom, ident, yr2))

    url_df = pd.DataFrame(rows_with_time, columns=["source","domain","id","year"])
    url_df.to_csv(os.path.join(OUTDIR, "urls_labeled_over_time.tsv"), sep="\t", index=False)

    def is_foreign(d): return d not in INTERNAL_DOMAINS
    foreign_df = url_df[url_df["domain"].map(is_foreign)]
    foreign_counts = foreign_df.groupby("source").size().rename("foreign_url_count")
    foreign_counts.to_csv(os.path.join(OUTDIR, "foreign_url_counts.tsv"), sep="\t")

    total_foreign_urls = len(foreign_df)
    print(f"Total foreign URLs: {total_foreign_urls}")
    with open(os.path.join(OUTDIR, "total_foreign_urls.txt"), "w") as f:
        f.write(f"Total foreign URLs in sample: {total_foreign_urls}\n")

    blob_to_langs = parse_blob_files(BLOB_FILES_TSV)
    prog_multi = sum(1 for langs in blob_to_langs.values() if len(langs) > 1)
    nl_multi_count = sum(1 for text in blobs_text.values() if len(classify_script_mix(text)) > 1)

    print("\n=== Summary ===")
    print(f"Total blobs analyzed: {total_sampled:,}")
    print(f"Text-like blobs:      {textlike:,}")
    print(f"Blobs with URLs:      {len(blob_has_url):,}")
    print("Top 10 URL domains:")
    for dom, count in url_domains_blob.most_common(10):
        print(f"  {dom}: {count}")
    print(f"Programming multi-language blobs: {prog_multi}")
    print(f"Natural-language multi-script blobs: {nl_multi_count}")
    print("================\n")

    with open(os.path.join(OUTDIR, "text_blob_count.txt"), "w") as f:
        f.write(f"Text-like blobs in sample: {textlike} of {total_sampled}\n")

    pd.Series(url_domains_blob).to_csv(os.path.join(OUTDIR, "url_domains.tsv"), sep="\t")
    with open(os.path.join(OUTDIR, "multilang_summary.txt"), "w") as out:
        out.write(f"Programming multi-language blobs: {prog_multi}\n")
        out.write(f"Natural-language multi-script blobs: {nl_multi_count}\n")

    df = pd.read_csv(os.path.join(OUTDIR, "url_domains.tsv"), sep="\t", header=None, names=["domain","count"])
    df = df.sort_values("count", ascending=False).head(20)
    fig = df.plot(kind="bar", x="domain", y="count", legend=False,
                  title="Top URL Domains").get_figure()
    fig.savefig(os.path.join(OUTDIR, "trace_top_url_domains.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()