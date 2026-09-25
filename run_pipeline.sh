#!/usr/bin/env bash
set -e

echo "=== StereoQueerEval & Toxic Training Pipeline Setup ==="

# 1. Install dependencies
if command -v pip &> /dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "Note: pip is not found in PATH. Please run inside your virtualenv or conda environment."
fi

# 2. Generate sample data if data folder is missing
if [ ! -d "data" ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    echo "Creating sample dataset..."
    python3 generate_sample_data.py
fi

# 3. Quick test run with Transformer
echo "Running quick smoke test with scratch Transformer (1 epoch)..."
python3 train.py --embed_source scratch --model transformer --epochs 1 --batch_size 16 --output_dir checkpoints_test

echo "Pipeline verified successfully!"
