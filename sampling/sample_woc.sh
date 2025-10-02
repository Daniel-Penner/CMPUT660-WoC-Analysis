#!/bin/bash

set -euo pipefail

OUTDIR=/data/play/$USER/sampling/samples
mkdir -p "$OUTDIR"

# Sample 1/1000 edges from relationship
p=0.001

log(){ printf "[%(%F %T)T] %s\n" -1 "$*" >&2; }

sample_relation(){
  rel=$1
  out="$OUTDIR/${rel}SampleU.s.gz"
  log ">>> Sampling $rel → $out"
  zcat /da?_data/basemaps/gz/${rel}FullU*.s \
    | awk -v p="$p" -v rel=$rel 'BEGIN{srand(12345)}
        NR % 100000000 == 0 {
          print strftime("[%F %T]"), "Processed", NR, "lines for", rel > "/dev/stderr"
        }
        { if (rand() < p) print }' \
    | gzip > "$out"
  log "<<< Finished $rel"
}

sample_relation c2dat
sample_relation c2P
sample_relation A2c