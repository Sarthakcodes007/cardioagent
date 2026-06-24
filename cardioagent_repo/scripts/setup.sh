#!/bin/bash
# CardioAgent — Complete RunPod Setup
# Run ONCE after pod creation: bash setup.sh
# PyTorch 2.4.0 + CUDA 12.4.1 already in template — do NOT reinstall

set -e
echo "========================================"
echo "CardioAgent — RunPod Setup"
echo "========================================"

# ── Python packages ──────────────────────────────────────────────────────────
echo "Installing packages..."
pip install --upgrade pip --quiet
pip install wfdb scikit-learn tqdm pandas matplotlib seaborn --quiet
pip install transformers accelerate sentencepiece --quiet
pip install sentence-transformers faiss-cpu --quiet
pip install langchain langchain-community pypdf --quiet
pip install onnx onnxruntime --quiet
pip install mistralai --quiet 2>/dev/null || true   # optional

echo "Packages installed."

# ── Directory structure ───────────────────────────────────────────────────────
echo "Creating directories..."
mkdir -p /workspace/cardioagent/data/mitdb
mkdir -p /workspace/cardioagent/data/nstdb
mkdir -p /workspace/cardioagent/data/ptbxl
mkdir -p /workspace/cardioagent/data/ltafdb
mkdir -p /workspace/cardioagent/data/chapman
mkdir -p /workspace/cardioagent/models
mkdir -p /workspace/cardioagent/results
mkdir -p /workspace/cardioagent/scripts

# Copy all scripts to scripts folder for backup
cp /workspace/*.py /workspace/cardioagent/scripts/ 2>/dev/null || true

echo "Directories created."

# ── Check existing data ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Checking existing datasets..."
echo "========================================"

# PTB-XL
PTBXL_COUNT=$(ls /workspace/cardioagent/data/ptbxl/records100/ 2>/dev/null | wc -l || echo 0)
if [ "$PTBXL_COUNT" -gt 20 ]; then
    echo "PTB-XL: FOUND ($PTBXL_COUNT subdirs)"
else
    echo "PTB-XL: MISSING — run download_data.sh"
fi

# LTAF
LTAF_COUNT=$(ls /workspace/cardioagent/data/ltafdb/ 2>/dev/null | grep .hea | wc -l || echo 0)
if [ "$LTAF_COUNT" -gt 50 ]; then
    echo "LTAF: FOUND ($LTAF_COUNT .hea files)"
else
    echo "LTAF: MISSING — run download_data.sh"
fi

# MIT-BIH
MITDB_COUNT=$(ls /workspace/cardioagent/data/mitdb/ 2>/dev/null | grep .hea | wc -l || echo 0)
if [ "$MITDB_COUNT" -gt 40 ]; then
    echo "MIT-BIH: FOUND"
else
    echo "MIT-BIH: MISSING — run download_data.sh"
fi

# Models
echo ""
echo "Checking trained models..."
for model in tier1_sqa.pt tier2_classifier.pt tier3_aggregator.pt; do
    if [ -f "/workspace/cardioagent/models/$model" ]; then
        echo "  $model: FOUND"
    else
        echo "  $model: MISSING — needs training"
    fi
done

echo ""
echo "========================================"
echo "Setup complete."
echo "If data is missing: bash download_data.sh"
echo "If models are missing: run tier1/2/3/4 scripts"
echo "========================================"
