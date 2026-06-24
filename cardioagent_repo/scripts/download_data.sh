#!/bin/bash
# CardioAgent — Data Download Script
# Only run if data is missing (check with setup.sh first)
# Uses wget instead of wfdb.dl_database() — lessons learned

PTBXL_DIR="/workspace/cardioagent/data/ptbxl"
LTAF_DIR="/workspace/cardioagent/data/ltafdb"
MITDB_DIR="/workspace/cardioagent/data/mitdb"
NSTDB_DIR="/workspace/cardioagent/data/nstdb"
CHAP_DIR="/workspace/cardioagent/data/chapman"

WGET="wget -q --show-progress --timeout=30 --tries=5"

# ── PTB-XL ───────────────────────────────────────────────────────────────────
PTBXL_COUNT=$(ls $PTBXL_DIR/records100/ 2>/dev/null | wc -l || echo 0)
if [ "$PTBXL_COUNT" -gt 20 ]; then
    echo "PTB-XL already downloaded, skipping."
else
    echo "Downloading PTB-XL (~520 MB, ~20 min)..."
    mkdir -p $PTBXL_DIR
    $WGET -O $PTBXL_DIR/ptbxl_database.csv \
      https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv
    $WGET -O $PTBXL_DIR/scp_statements.csv \
      https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
    wget -r -N -c -np -nH --cut-dirs=3 \
      --timeout=30 --tries=5 \
      --show-progress \
      -P $PTBXL_DIR \
      https://physionet.org/files/ptb-xl/1.0.3/records100/
    echo "PTB-XL done."
fi

# ── MIT-BIH + NSTDB ──────────────────────────────────────────────────────────
MITDB_COUNT=$(ls $MITDB_DIR/ 2>/dev/null | grep .hea | wc -l || echo 0)
if [ "$MITDB_COUNT" -gt 40 ]; then
    echo "MIT-BIH already downloaded, skipping."
else
    echo "Downloading MIT-BIH (~100 MB)..."
    mkdir -p $MITDB_DIR
    wget -r -N -c -np -nH --cut-dirs=3 \
      --timeout=30 --tries=5 -q \
      -P $MITDB_DIR \
      https://physionet.org/files/mitdb/1.0.0/
    echo "MIT-BIH done."
fi

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

# ── LTAF ─────────────────────────────────────────────────────────────────────
LTAF_COUNT=$(ls $LTAF_DIR/ 2>/dev/null | grep .hea | wc -l || echo 0)
if [ "$LTAF_COUNT" -gt 50 ]; then
    echo "LTAF already downloaded, skipping."
else
    echo "Downloading LTAF (~1.5 GB, ~30 min)..."
    mkdir -p $LTAF_DIR
    wget -r -N -c -np -nH --cut-dirs=3 \
      --timeout=30 --tries=5 \
      --show-progress \
      -P $LTAF_DIR \
      https://physionet.org/files/ltafdb/1.0.0/
    echo "LTAF done."
fi

# ── Chapman-Shaoxing ──────────────────────────────────────────────────────────
CHAP_COUNT=$(ls $CHAP_DIR/ 2>/dev/null | grep .hea | wc -l || echo 0)
if [ "$CHAP_COUNT" -gt 100 ]; then
    echo "Chapman already downloaded, skipping."
else
    echo "Downloading Chapman-Shaoxing (~1.5 GB, ~20 min)..."
    mkdir -p $CHAP_DIR
    wget -r -N -c -np -nH --cut-dirs=3 \
      --timeout=30 --tries=5 \
      --show-progress \
      -P $CHAP_DIR \
      https://physionet.org/files/ecg-arrhythmia/1.0.0/
    echo "Chapman done."
fi

echo ""
echo "All downloads complete."
echo "Run: bash setup.sh to verify."
