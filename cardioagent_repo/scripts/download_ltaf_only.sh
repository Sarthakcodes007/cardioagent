#!/bin/bash
# CardioAgent — LTAF-ONLY download
# This rerun (Tier 3 attention pooling + Ablation hard negatives) needs
# ONLY the LTAF Database. No PTB-XL, no MIT-BIH, no Chapman.

LTAF_DIR="/workspace/cardioagent/data/ltafdb"

LTAF_COUNT=$(ls $LTAF_DIR/ 2>/dev/null | grep .hea | wc -l || echo 0)
if [ "$LTAF_COUNT" -gt 50 ]; then
    echo "LTAF already downloaded ($LTAF_COUNT records), skipping."
else
    echo "Downloading LTAF (~1.5 GB, ~30-45 min)..."
    mkdir -p $LTAF_DIR
    wget -r -N -c -np -nH --cut-dirs=3 \
      --timeout=30 --tries=5 \
      --show-progress \
      -P $LTAF_DIR \
      https://physionet.org/files/ltafdb/1.0.0/
    echo "LTAF done."
fi

echo ""
echo "Ready. Only LTAF downloaded — that's all this rerun needs."
