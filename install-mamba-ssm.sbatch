#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:20:00
#SBATCH --job-name=install-mamba-ssm
#SBATCH --output=logs/install-mamba-ssm-%j.log
#SBATCH --error=logs/install-mamba-ssm-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --environment=alps3
#SBATCH --no-requeue

mkdir -p logs

echo "[$(date)] Python: $(which python) $(python --version)"
echo "[$(date)] Torch location: $(python -c 'import torch; print(torch.__file__)')"

# Make torch findable by the build subprocess (needed for mamba-ssm's setup.py)
TORCH_DIR=$(python -c "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
export PYTHONPATH=$TORCH_DIR:$PYTHONPATH

# Ensure CUDA toolkit headers are on the path for CUDA extension compilation
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

echo "[$(date)] PYTHONPATH=$PYTHONPATH"
echo "[$(date)] CUDA_HOME=$CUDA_HOME"

echo "[$(date)] Installing causal-conv1d (mamba-ssm dependency)..."
pip install causal-conv1d --no-build-isolation --user

echo "[$(date)] Installing mamba-ssm..."
pip install mamba-ssm --no-build-isolation --user

echo "[$(date)] Verifying install..."
python -c "import mamba_ssm; print('mamba-ssm version:', mamba_ssm.__version__)"
echo "[$(date)] Done."
