#!/bin/bash
set -euo pipefail

OUTDIR=/data/play/dpenner/size_metrics
OUTFILE="$OUTDIR/size_metrics.txt"

# During the running of this file da3 was down and attempting to iterate through da3_data was causing crashes.
SERVERS="da0 da1 da2 da4 da5"

exec >"$OUTFILE"

stream_relation() {
  local rel=$1
  for srv in $SERVERS; do
    if ls /${srv}_data/basemaps/gz/${rel}FullU*.s.gz >/dev/null 2>&1; then
      log "Streaming $rel from $srv"
      zcat /${srv}_data/basemaps/gz/${rel}FullU*.s.gz 2>/dev/null || true
    else
      log "Skipping $srv (no $rel data)"
    fi
  done
}

# 1. Number of unique commits
log "Counting commits..."
echo "[1] Number of commits (c2dat):"
stream_relation c2dat \
  | awk 'NR % 10000000 == 0 {print NR, "lines processed for commits" > "/dev/stderr"; fflush("/dev/stderr")} {print}' \
  | wc -l
echo

# 2. Number of unique deforked projects
log "Counting unique projects..."
echo "[2] Number of unique projects (c2P):"
stream_relation c2P \
  | awk 'NR % 10000000 == 0 {print NR, "lines processed for projects" > "/dev/stderr"; fflush("/dev/stderr")} {print}' \
  | cut -d';' -f2 | sort -u | wc -l
echo

# 3. Number of uniqued aliased authors
log "Counting unique authors..."
echo "[3] Number of unique authors (A2c):"
stream_relation A2c \
  | awk 'NR % 10000000 == 0 {print NR, "lines processed for authors" > "/dev/stderr"; fflush("/dev/stderr")} {print}' \
  | cut -d';' -f1 | sort -u | wc -l
echo