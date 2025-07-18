#!/bin/bash
#SBATCH --job-name=bicubic_eval_fair
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00

echo "============================================"
echo "Bicubic Evaluation (Fair Comparison) Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Setup environment
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Project directories
PROJECT_DIR=$HOME/bicubic_evaluation
NEURAL_NET_DIR=$HOME/2x_temperature_sr_project
DATA_DIR=$HOME/temperature_sr_project/data

cd $PROJECT_DIR

# Install dependencies
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $NEURAL_NET_DIR:$NEURAL_NET_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user torch torchvision opencv-python tqdm matplotlib scipy pyyaml

# Verify environment and imports
echo "Verifying environment and neural network imports..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $NEURAL_NET_DIR:$NEURAL_NET_DIR \
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

# Test neural network imports
print('Testing neural network imports...')
sys.path.append('$NEURAL_NET_DIR')
try:
    from data_preprocessing import TemperatureDataPreprocessor
    print('✅ Successfully imported TemperatureDataPreprocessor')
except ImportError as e:
    print(f'❌ Failed to import TemperatureDataPreprocessor: {e}')
"

echo "============================================"
echo "Starting bicubic evaluation with neural network preprocessing..."

# **CRITICAL**: Add bind for neural network project directory
# Run evaluation with all necessary binds
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $NEURAL_NET_DIR:$NEURAL_NET_DIR \
    --bind $DATA_DIR:$DATA_DIR \
    --env PYTHONPATH=$PROJECT_DIR:$NEURAL_NET_DIR:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python bicubic_evaluation.py

echo "============================================"
echo "Bicubic Evaluation Job Finished: $(date)"
echo "============================================"