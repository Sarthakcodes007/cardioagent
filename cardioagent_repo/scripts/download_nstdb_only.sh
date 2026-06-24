#!/bin/bash
# NSTDB-only download (~5MB) — needed for real noise injection in the
# real-model ablation rerun. Quick add to the existing LTAF-only pod.

NSTDB_DIR="/workspace/cardioagent/data/nstdb"
NSTDB_COUNT=$(ls $NSTDB_DIR/ 2>/dev/null | grep .hea | wc -l || echo 0)
if [ "$NSTDB_COUNT" -gt 2 ]; then
    echo "NSTDB already downloaded, skipping."
else
    echo "Downloading NSTDB (~5 MB)..."
    mkdir -p $NSTDB_DIR
    wget -r -N -c -np -nH --cut-dirs=3 \
      --timeout=30 --tries=5 -q \
      -P $NSTDB_DIR \
      https://physionet.org/files/nstdb/1.0.0/
    echo "NSTDB done."
fi
