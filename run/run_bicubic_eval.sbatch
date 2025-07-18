#!/bin/bash
#SBATCH --job-name=bicubic_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00

echo "============================================"
echo "Bicubic Evaluation Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Setup environment
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Project directory
PROJECT_DIR=$HOME/bicubic_evaluation
cd $PROJECT_DIR

# Install dependencies
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user torch torchvision opencv-python tqdm matplotlib scipy pyyaml

# Verify environment
echo "Verifying environment..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import cv2
print(f'OpenCV: {cv2.__version__}')
import numpy as np
print(f'NumPy: {np.__version__}')
"

echo "============================================"
echo "Starting bicubic evaluation..."

# Mount data directory
DATA_DIR=$HOME/temperature_sr_project/data

# Run evaluation
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $DATA_DIR:$DATA_DIR \
    --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python bicubic_evaluation.py

echo "============================================"
echo "Bicubic Evaluation Job Finished: $(date)"
echo "============================================"