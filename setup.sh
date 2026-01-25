#!/bin/bash
# RLVR Setup Script for NVIDIA GB10 / DGX Spark
# Based on Matthew Berman's tutorial

set -e

echo "=== RLVR Environment Setup ==="

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^RL "; then
    echo "Creating conda environment 'RL'..."
    conda create -n RL python=3.11 -y
fi

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RL

echo "Installing PyTorch with CUDA 13.0..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo "Installing transformers and dependencies..."
pip install transformers==4.56.2 datasets accelerate peft bitsandbytes

echo "Installing Unsloth, Unsloth Zoo, and TRL..."
pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth" --no-deps trl==0.22.2

echo "Installing Jupyter and TensorBoard..."
pip install jupyter ipykernel tensorboard

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=RL --display-name="Python (RL)"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate RL"
echo "Start Jupyter: jupyter lab"
