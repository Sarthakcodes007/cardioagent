#!/bin/bash
# Run this ON THE RUNPOD POD to bundle everything needed to avoid
# retraining from scratch next time. Then download the zip via the
# RunPod file browser.

echo "Creating backup zip of all models, results, and figures..."

cd /workspace
mkdir -p /workspace/cardioagent_backup

# Trained model checkpoints (the expensive part to redo)
mkdir -p /workspace/cardioagent_backup/models
cp /workspace/cardioagent/models/*.pt /workspace/cardioagent_backup/models/ 2>/dev/null
cp /workspace/cardioagent/models/*.onnx /workspace/cardioagent_backup/models/ 2>/dev/null

# All results JSONs
mkdir -p /workspace/cardioagent_backup/results
cp /workspace/cardioagent/results/*.json /workspace/cardioagent_backup/results/ 2>/dev/null
cp /workspace/cardioagent/*.json /workspace/cardioagent_backup/results/ 2>/dev/null

# All generated figures (PNG + PDF)
mkdir -p /workspace/cardioagent_backup/figures
cp /workspace/cardioagent/results/*.png /workspace/cardioagent_backup/figures/ 2>/dev/null
cp /workspace/cardioagent/results/*.pdf /workspace/cardioagent_backup/figures/ 2>/dev/null

# All scripts (so the pipeline is fully reproducible from this snapshot)
mkdir -p /workspace/cardioagent_backup/scripts
cp /workspace/*.py /workspace/cardioagent_backup/scripts/ 2>/dev/null
cp /workspace/*.sh /workspace/cardioagent_backup/scripts/ 2>/dev/null

echo ""
echo "Contents being backed up:"
find /workspace/cardioagent_backup -type f | sort

cd /workspace
zip -r cardioagent_full_backup.zip cardioagent_backup/

echo ""
echo "=========================================="
echo "Done. Download this file via RunPod's file browser:"
echo "  /workspace/cardioagent_full_backup.zip"
echo "=========================================="
ls -lh /workspace/cardioagent_full_backup.zip
