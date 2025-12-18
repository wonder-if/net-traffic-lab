#!/usr/bin/env bash
# Generate all artifacts required by interactive_app.py
# Assumes data present locally:
#   - USTC traffic CSVs under ustc_all/
#   - IDS2017 CSV under ids2017/data.csv
# Uses CPU by default; tweak flags as needed.

set -euo pipefail

echo "Running semi-supervised sweeps on USTC data..."
python semi_supervised_experiment.py \
  --max-per-class 1000 \
  --test-per-class 200 \
  --unlabeled-cap-per-class 600 \
  --epochs 8 \
  --labeled-ratios 0.01,0.02,0.05,0.1,0.2,0.5

echo "Running IDS2017 feature stability analysis..."
python ids2017_feature_stability.py \
  --data-path ids2017/data.csv \
  --max-rows 50000 \
  --n-estimators 150 \
  --seeds 0,1,2 \
  --test-size 0.25

echo "Done. Outputs saved in outputs/ and ids2017_feature_outputs/ for the Streamlit app."
